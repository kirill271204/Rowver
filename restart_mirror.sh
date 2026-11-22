#!/bin/bash
# restart the Isaac live mirror (MuJoCo keeps running). Extra args go to isaac_live.py, e.g. --fog 0
docker exec vscode pkill -f "tools/isaac_live.py" 2>/dev/null
sleep 3
nohup docker exec -u ubuntu:ubuntu -w /isaac-sim vscode /isaac-sim/python.sh /tmp/rowver_tools/isaac_live.py \
    --dir /tmp/rowver_live "$@" > ~/isaac_live.log 2>&1 &
echo "mirror restarting (log: ~/isaac_live.log)"
