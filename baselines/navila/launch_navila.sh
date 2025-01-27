#!/bin/bash

# Create a new tmux session
session_name="server_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane

# Launch the VLM server
tmux select-pane -t 0
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "python navila_policy.py $1" Enter

# Launch the 

# Attach to the tmux session
tmux -2 attach-session -t $session_name
