#!/bin/bash

# Create a new tmux session
session_name="task_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 2    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves


# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "source ~/create_ws/install/setup.bash" Enter
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "launch/launch_robot_wout_lidar.sh" Enter  

# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 1
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "python navigate_vla.py $@" Enter

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 2
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "python pd_controller.py" Enter

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 3
tmux send-keys "ros2 launch foxglove_bridge foxglove_bridge_launch.xml" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
