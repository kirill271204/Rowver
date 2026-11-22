#!/bin/bash
# run_mission.sh <run-name> [run_fleet args...]: stop any running mission, start this one headless in the
# background (MUJOCO_GL=egl), log to ~/<run-name>.log. Add --live --record to mirror it to Isaac and record it.
name=$1; shift
for p in $(pgrep -f "m rowver.run_fleet"); do kill "$p"; done
sleep 2
cd ~/rowver && MUJOCO_GL=egl WINDIR=$HOME/winfonts nohup .venv/bin/python -u -m rowver.run_fleet --fresh \
    --run-name "$name" "$@" > ~/"$name".log 2>&1 &
echo "mission $name started: $* (log ~/$name.log)"
