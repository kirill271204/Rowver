"""Thalamus: the contract between a body and the brain ("nerves").

A body (simulated or real) exposes four calls, at about 2 Hz (once per skill execution):
  describe()             robot_id, body descriptors, skills (action bounds, features, targets)
  reset()                start a new episode
  context(skill, goal)   the state features the brain decides on (sensors + goal + reference)
  act(skill, action)     execute one skill step; returns the measured outcome
The body runs its own fast loop (sensors -> spinal-cord controller -> actuators, 500 Hz) and
never waits on the brain inside that loop.

Transports, same interface:
  LocalNerve   direct calls in one process (fast practice in simulation)
  UDPNerve     JSON datagrams with sequence numbers, ack and retry (separate processes/machines);
               serve_udp() puts any body behind it
"""
from __future__ import annotations

import json
import socket
from dataclasses import asdict, dataclass, field

import numpy as np

from .provenance import Tagged


@dataclass
class SkillSchema:
    name: str
    family: str                           # skills in one family can share experience
    actions: dict[str, Tagged]            # name -> Tagged((lo, hi)): the correction bounds
    features: list[str]                   # context() keys the brain uses
    targets: list[str]                    # measured outcomes TabPFN predicts
    goal: dict[str, str]                  # target -> goal feature it should match
    minimize: list[str]                   # targets to keep small (e.g. tilt)
    fail: str = "fail"                    # binary failure label (delayed: fail during this or next steps)
    info: dict[str, Tagged] = field(default_factory=dict)   # other tagged numbers the skill uses

    def to_json(self):
        d = asdict(self)
        d["actions"] = {k: {"value": list(v.value), "source": v.source} for k, v in self.actions.items()}
        d["info"] = {k: {"value": _plain(v.value), "source": v.source} for k, v in self.info.items()}
        return d

    @staticmethod
    def from_json(d):
        d = dict(d)
        d["actions"] = {k: Tagged(tuple(v["value"]), v["source"]) for k, v in d["actions"].items()}
        d["info"] = {k: Tagged(v["value"], v["source"]) for k, v in d["info"].items()}
        return SkillSchema(**d)


class Body:
    """What a robot package implements (a body lives in its own package, never in brain/)."""
    robot_id: str
    descriptors: dict[str, Tagged]        # body-independent description: leg length, mass, ...
    skills: dict[str, SkillSchema]

    def reset(self): ...
    def context(self, skill: str, goal: dict) -> dict: ...
    def act(self, skill: str, action: dict) -> dict: ...


class LocalNerve:
    def __init__(self, body: Body):
        self.body = body

    def describe(self) -> dict:
        b = self.body
        return {"robot_id": b.robot_id, "descriptors": b.descriptors, "skills": b.skills}

    def reset(self):
        return self.body.reset()

    def context(self, skill, goal):
        return self.body.context(skill, goal)

    def act(self, skill, action):
        return self.body.act(skill, action)

    def commit(self, skill, action, pred, out, n_done):
        """The body's offer to drive the same action on without a new plan: a context, or None."""
        f = getattr(self.body, "commit", None)
        return f(skill, action, pred, out, n_done) if f is not None else None

    def counterfactual(self, skill, actions):
        """Simulation only: outcomes of actions NOT taken, from the current moment (the body
        restores the world after each); [] when the body cannot."""
        f = getattr(self.body, "counterfactual", None)
        return f(skill, actions) if f is not None else []


# ------------------------------------------------------------------ UDP transport
MAX_DGRAM = 65000


def _plain(o):
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


class UDPNerve:
    """Client side. Each request carries a sequence number; the reply echoes it. Lost datagrams
    are retried; a late duplicate reply is ignored."""

    def __init__(self, host="127.0.0.1", port=47100, timeout=30.0, retries=3):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        self.retries = retries
        self.seq = 0

    def _call(self, op, **kw):
        self.seq += 1
        msg = json.dumps({"seq": self.seq, "op": op, **_plain(kw)}).encode()
        for _ in range(self.retries + 1):
            self.sock.sendto(msg, self.addr)
            try:
                while True:
                    data, _ = self.sock.recvfrom(MAX_DGRAM)
                    rep = json.loads(data)
                    if rep.get("seq") == self.seq:
                        if "error" in rep:
                            raise RuntimeError(f"body error in {op}: {rep['error']}")
                        return rep["result"]
            except socket.timeout:
                continue
        raise TimeoutError(f"no reply from body at {self.addr} for {op}")

    def describe(self):
        d = self._call("describe")
        return {"robot_id": d["robot_id"],
                "descriptors": {k: Tagged(v["value"], v["source"]) for k, v in d["descriptors"].items()},
                "skills": {k: SkillSchema.from_json(v) for k, v in d["skills"].items()}}

    def reset(self):
        return self._call("reset")

    def context(self, skill, goal):
        return self._call("context", skill=skill, goal=goal)

    def act(self, skill, action):
        return self._call("act", skill=skill, action=action)


def serve_udp(body: Body, host="127.0.0.1", port=47100):
    """Body side: answer UDPNerve requests forever. A repeated sequence number (the client
    retried) gets the cached reply instead of executing the action twice."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    last_seq, last_reply = None, None
    print(f"body {body.robot_id} listening on udp://{host}:{port}", flush=True)
    while True:
        data, addr = sock.recvfrom(MAX_DGRAM)
        req = json.loads(data)
        if req["seq"] == last_seq:
            sock.sendto(last_reply, addr)
            continue
        try:
            op = req["op"]
            if op == "describe":
                res = {"robot_id": body.robot_id,
                       "descriptors": {k: {"value": _plain(v.value), "source": v.source} for k, v in body.descriptors.items()},
                       "skills": {k: s.to_json() for k, s in body.skills.items()}}
            elif op == "reset":
                res = body.reset()
            elif op == "context":
                res = body.context(req["skill"], req["goal"])
            elif op == "act":
                res = body.act(req["skill"], req["action"])
            else:
                raise ValueError(f"unknown op {op}")
            rep = {"seq": req["seq"], "result": _plain(res)}
        except Exception as e:  # report the error to the brain instead of dying
            rep = {"seq": req["seq"], "error": f"{type(e).__name__}: {e}"}
        last_seq, last_reply = req["seq"], json.dumps(rep).encode()
        sock.sendto(last_reply, addr)
