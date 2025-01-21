#! /bin/bash

# Create a new tmux session
session_name="robot_launch_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0 
tmux splitw -v -p 50
tmux selectp -t 2
tmux splitw -v -p 50

# Launch the camera
tmux select-pane -t 0
tmux send-keys "conda activate lcbc" Enter
tmux send-keys "python propose_counterfactuals.py --data_dir /home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets_backup/sacson_labelled --config counterfactuals_sacson.yaml" Enter

# Launch the joy controller
tmux select-pane -t 1
tmux send-keys "conda activate lcbc" Enter
tmux send-keys "python propose_counterfactuals.py --data_dir /home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets_backup/scand_labelled --config counterfactuals_scand.yaml" Enter

# Launch lidar
tmux select-pane -t 2
tmux send-keys "conda activate lcbc" Enter
tmux send-keys "python propose_counterfactuals.py --data_dir /home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets_backup/cory_hall_labelled --config counterfactuals_cory_hall.yaml" Enter

# Publish static transform
tmux select-pane -t 3
tmux send-keys "conda activate lcbc" Enter
tmux send-keys "python propose_counterfactuals.py --data_dir /home/noam/LLLwL/lcbc/data/data_annotation/lcbc_datasets_backup/go_stanford_cropped_labelled --config counterfactuals_go_stanford.yaml" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name