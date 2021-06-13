#!/bin/bash

cd ~

gnome-terminal -t LEO -- bash -c "source sim_ws/devel/setup.bash; roslaunch leo_gazebo leo_marsyard.launch"
sleep 8
gnome-terminal -t pose_relay -- bash -c "python3 ~/Desktop/SAFARI/scripts/pose_relay.py"
gnome-terminal -t "SAFARI LAUNCH" -- bash -c "source catkin_ws/devel/setup.bash; cd ~/Desktop/SAFARI/launch; roslaunch main.launch"
sleep 10
gnome-terminal -t "SAFARI AI" -- bash -c "cd ~/Desktop/SAFARI; python3 main.py"
