"""The brain's one model: TabPFN v3.5-fast, on this GPU or on Prior Labs' servers.

CORTEX_TABPFN_BACKEND (.env):
  local   weights tabpfn-v3.5-fast-20260909.safetensors from huggingface.co/Prior-Labs/tabpfn_3_5
          (not gated), cached in TabPFN's cache folder (%APPDATA%\\tabpfn); `python -m
          brain.check_tabpfn` downloads it once. Offline after that.
  cloud   the Prior Labs API (tabpfn_client), the same v3.5-fast model, always fitted with
          fit_mode="fit_with_cache": the server keeps the context's KV cache under the fit's id, so
          every later predict against that context is one request that skips re-encoding the rows.
          Needs TABPFN_TOKEN (an API key; keep it in .env, never in code).
          The API allows 60 predict requests a minute: every cloud call takes a slot from a shared
          per-minute bucket (CORTEX_CLOUD_PREDICTS_PER_MIN, default 56; CORTEX_CLOUD_FITS_PER_MIN,
          default 30), and an HTTP 429 waits the time the server asks for.
There is no fallback to another version. Other settings: CORTEX_TABPFN_DEVICE (auto, local only),
CORTEX_TABPFN_N_ESTIMATORS.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=".*")  # TabPFN is chatty on tiny tables
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
# Local backend: every fitted model shares ONE built network per task (regressor, classifier) by
# reference. Inference never changes the weights, and without this each fit holds its own 0.34 GB
# copy on the GPU: fit_cache keeps up to 64 fits, and Windows spills the overflow into system RAM
# until the process dies (2026-11-22, a 3-rover run crashed after ~10 refits).
os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "2")

MODEL_VERSION = "v3.5-fast"
HF_REPO = "Prior-Labs/tabpfn_3_5"
QUANTILES = np.round(np.linspace(0.02, 0.98, 25), 4)


class TabPFNSetupError(RuntimeError):
    pass


def device() -> str:
    return os.getenv("CORTEX_TABPFN_DEVICE", "auto")


def backend() -> str:
    b = os.getenv("CORTEX_TABPFN_BACKEND", "local").strip().lower()
    if b not in ("local", "cloud"):
        raise TabPFNSetupError(f"CORTEX_TABPFN_BACKEND must be 'local' or 'cloud', not {b!r}")
    return b


_CLOUD_READY = False


def _cloud():
    """Authenticate the Prior Labs client once (token from TABPFN_TOKEN)."""
    global _CLOUD_READY
    import tabpfn_client
    if not _CLOUD_READY:
        tok = os.getenv("TABPFN_TOKEN", "").strip()
        if not tok:
            raise TabPFNSetupError("CORTEX_TABPFN_BACKEND=cloud needs TABPFN_TOKEN in .env "
                                   "(an API key from https://platform.priorlabs.ai/account/api-keys)")
        tabpfn_client.set_access_token(tok)
        _CLOUD_READY = True
    return tabpfn_client


class _Bucket:
    """At most `n` calls in any 60 s window, shared by every thread (the API's per-minute limit)."""

    def __init__(self, n):
        import threading
        from collections import deque
        self.n, self.t, self.lock = int(n), deque(), threading.Lock()
        self.waited = 0.0

    def take(self):
        while True:
            with self.lock:
                now = time.monotonic()
                while self.t and now - self.t[0] >= 60.0:
                    self.t.popleft()
                if len(self.t) < self.n:
                    self.t.append(now)
                    return
                dt = 60.0 - (now - self.t[0]) + 0.05
            self.waited += dt
            time.sleep(dt)


_BUCKETS: dict = {}


def _bucket(kind):
    if kind not in _BUCKETS:
        env = {"predict": ("CORTEX_CLOUD_PREDICTS_PER_MIN", "56"), "fit": ("CORTEX_CLOUD_FITS_PER_MIN", "30")}[kind]
        _BUCKETS[kind] = _Bucket(os.getenv(*env))
    return _BUCKETS[kind]


def cloud_waited() -> float:
    """Seconds spent waiting for the API's per-minute request limits so far."""
    return sum(b.waited for b in _BUCKETS.values())


COUNTS = {"fit": 0, "predict": 0, "lost_fits": 0, "retries": 0}   # requests sent to the Prior Labs API


class LostFit(RuntimeError):
    """The server no longer has (or can no longer use) a fit: fit it again (fit_cache does)."""


def _http_status(msg: str):
    import re
    m = re.search(r"\[HTTP (\d{3})\]", msg)
    return int(m.group(1)) if m else None


class CloudModel:
    """A tabpfn_client model behind the API's per-minute limits. Errors by class:
      HTTP 429                              wait the time the server asks for, retry (never refit)
      408/502/503/504, timeouts, transport  retry with backoff (up to 6 attempts), then raise
      HTTP 500                              retry twice, then raise
      FittedModelNotFoundError,
      RefitRequiredError                    raise LostFit (the fit cache refits once)
      401/403 and anything else             raise at once (an auth error is never taken for a lost fit)"""

    def __init__(self, inner):
        self.inner = inner

    def _call(self, kind, fn, *a, **kw):
        import re
        import httpx
        from tabpfn_client.errors import FittedModelNotFoundError, RefitRequiredError, RetryableServerError
        n500 = 0
        for attempt in range(6):
            _bucket(kind).take()
            COUNTS[kind] += 1
            try:
                return fn(*a, **kw)
            except (FittedModelNotFoundError, RefitRequiredError) as e:
                COUNTS["lost_fits"] += 1
                raise LostFit(str(e)) from e
            except (RetryableServerError, httpx.TimeoutException, httpx.TransportError) as e:
                if attempt == 5:
                    raise
                dt = min(2.0 ** attempt, 20.0)
                err = e
            except RuntimeError as e:
                msg, code = str(e), _http_status(str(e))
                if code == 429 and attempt < 5:
                    m = re.search(r"Retry in (\d+(?:\.\d+)?)s", msg)
                    dt = float(m.group(1)) + 0.5 if m else 10.0
                elif code == 500 and n500 < 2 and attempt < 5:
                    n500 += 1
                    dt = 3.0 * n500
                else:
                    raise
                err = e
            COUNTS["retries"] += 1
            _bucket(kind).waited += dt
            time.sleep(dt)
        raise err

    def fit(self, X, y):
        self._call("fit", self.inner.fit, X, y)
        return self

    def predict(self, X, **kw):
        return self._call("predict", self.inner.predict, X, **kw)

    def predict_proba(self, X):
        return self._call("predict", self.inner.predict_proba, X)

    # ---- persistence of the server-side fit (fit_cache)
    def record(self) -> dict:
        return self.inner.save_model()

    @classmethod
    def from_record(cls, kind: str, record: dict) -> "CloudModel":
        """A fitted model from a save_model() record: no request is made here."""
        tc = _cloud()
        est = tc.TabPFNRegressor if kind == "reg" else tc.TabPFNClassifier
        return cls(est.load_model(record))


def client_version() -> str:
    try:
        import importlib.metadata as md
        return md.version("tabpfn-client")
    except Exception:
        return "none"


def model_params(kind: str) -> dict:
    """Everything about the estimator that changes what a fit produces (part of the fit cache key)."""
    b = backend()
    return {"backend": b, "kind": kind, "model_version": MODEL_VERSION, "n_estimators": n_estimators(),
            "fit_mode": "fit_with_cache" if b == "cloud" else "fit_preprocessors", "random_state": 0,
            "client": client_version() if b == "cloud" else "local"}


def is_cloud(model) -> bool:
    return isinstance(model, CloudModel)


def n_estimators() -> int:
    return int(os.getenv("CORTEX_TABPFN_N_ESTIMATORS", "4"))


def weights_path() -> Path:
    from tabpfn.model_loading import ModelSource, prepend_cache_path
    return Path(prepend_cache_path(ModelSource.get_v3_5_fast().default_filename))


def download_weights() -> Path:
    p = weights_path()
    if not p.exists():
        from huggingface_hub import hf_hub_download
        print(f"downloading {p.name} from huggingface.co/{HF_REPO} ...", flush=True)
        hf_hub_download(HF_REPO, p.name, local_dir=p.parent)
    return p


_CLIENT = None          # a shared brain server's client (rowver/hive): make_model returns its proxies


def set_client(client):
    """Route every make_model() through a shared brain server (None: local TabPFN models again)."""
    global _CLIENT
    _CLIENT = client


def make_model(kind: str, fit_mode: str = "fit_preprocessors", local: bool = False, lineage=None):
    """kind: 'reg' | 'clf'. A fresh TabPFN v3.5-fast regressor or classifier (or, with a brain server
    installed, a proxy with the same fit / predict / predict_proba calls that the server batches).
    local=True always builds a real model of the configured backend, never a proxy (the brain
    server itself uses this). On the cloud backend every model is fitted with fit_with_cache.
    lineage: what this model predicts (e.g. (columns, target)); the brain server serves a new
    context's requests from the previous context of the same lineage while it fits the new one."""
    if _CLIENT is not None and not local:
        return _CLIENT.model(kind, lineage)
    if backend() == "cloud":
        tc = _cloud()
        cls = tc.TabPFNRegressor if kind == "reg" else tc.TabPFNClassifier
        return CloudModel(cls.create_default_for_version(MODEL_VERSION, n_estimators=n_estimators(),
                                                         fit_mode="fit_with_cache", random_state=0))
    if not weights_path().exists():
        raise TabPFNSetupError(f"TabPFN v3.5-fast weights not found at {weights_path()}. "
                               "Run once:  .venv/Scripts/python.exe -m brain.check_tabpfn  (downloads them, "
                               "no login or token); inference then stays local and offline.")
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion
    cls = TabPFNRegressor if kind == "reg" else TabPFNClassifier
    return cls.create_default_for_version(ModelVersion(MODEL_VERSION), device=device(),
                                          n_estimators=n_estimators(), fit_mode=fit_mode,
                                          random_state=0)


def backend_name() -> str:
    if backend() == "cloud":
        return f"TabPFN {MODEL_VERSION} (Prior Labs cloud, fit_with_cache KV cache, {n_estimators()} estimators)"
    return f"TabPFN {MODEL_VERSION} (local, {n_estimators()} estimators, device {device()})"


def sample_from_quantiles(qmat: np.ndarray, n_samples: int, rng) -> np.ndarray:
    """qmat (n, Q) at QUANTILES -> samples (n, S) via inverse-CDF interpolation."""
    qmat = np.sort(qmat, axis=1)
    n, Q = qmat.shape
    u = rng.uniform(QUANTILES[0], QUANTILES[-1], size=(n, n_samples))
    pos = (u - QUANTILES[0]) / (QUANTILES[1] - QUANTILES[0])
    i = np.clip(np.floor(pos).astype(int), 0, Q - 2)
    frac = pos - i
    lo = np.take_along_axis(qmat, i, axis=1)
    hi = np.take_along_axis(qmat, i + 1, axis=1)
    return lo + frac * (hi - lo)


def qcol(level: float) -> int:
    return int(np.argmin(np.abs(QUANTILES - level)))
