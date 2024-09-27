#!/bin/bash

# Create a new tmux session
session_name="robot_launch_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves

# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 0
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "ros2 run usb_cam usb_cam_node_exe /image_raw:=/front/image_raw params_file:=../../config/camera.yaml " Enter

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 1
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "ros2 launch teleop_twist_joy teleop-launch.py joy_config:=xbox" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
