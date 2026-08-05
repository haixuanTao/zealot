#!/bin/bash
# Launch the REAL GR00T manager streamer (button-driven, for driving sim/robot
# with the SONIC pipeline). For viz/VR-sim use pose_pub.py instead — they both
# bind :5556, so only one can run at a time.
fuser -k 5556/tcp 2>/dev/null; sleep 1
cd ~/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
rm -f /tmp/pico_server.log
setsid nohup python -u gear_sonic/scripts/pico_manager_thread_server.py --manager > /tmp/pico_server.log 2>&1 < /dev/null &
