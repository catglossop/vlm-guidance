#!/bin/bash

# Create a new tmux session
session_name="task_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
# tmux selectp -t 0    # select the first (0) pane
# tmux splitw -v -p 50 # split it into two halves

# tmux selectp -t 2    # select the new, second (2) pane
# tmux splitw -v -p 50 # split it into two halves
# tmux selectp -t 0    # go back to the first pane

# Run the roslaunch command in the first pane
# tmux select-pane -t 0
# tmux send-keys "source ~/create_ws/install/setup.bash" Enter
# tmux send-keys "conda deactivate" Enter
# tmux send-keys "ros2 launch deployment robot_launch.py" Enter  

# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 0
# tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "conda activate lifelong" Enter
tmux send-keys "python low_level_policy_unconditioned.py $@" Enter

# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 1
tmux send-keys "conda activate lifelong" Enter
tmux send-keys "python ../../../../deployment/deployment/pd_controller.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
