"""Cortex brain: robot-agnostic regions. No robot code lives here; robots import this package.

  nerves.py        thalamus: the Body/Nerve contract (describe, reset, context, act), local + UDP
  hippocampus.py   shared experience table for all robots and skills, recall()
  world_model.py   cerebellum: TabPFN outcome quantiles and P(fail) on recalled rows
  prefrontal.py    planner: chooses an action for a goal (CEM), reflex veto on P(fail)
  brain.py         the loop: sense -> recall -> predict -> decide -> act -> remember
  provenance.py    brain.toml loader; every number is tagged with its source
  tabpfn_model.py  the one model: TabPFN v3.5-fast, local and offline
  check_tabpfn.py  python -m brain.check_tabpfn: download the weights once, smoke check
"""
