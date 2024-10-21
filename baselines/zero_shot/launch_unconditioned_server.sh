#!/bin/bash

# Create a new tmux session
session_name="server_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves

# Launch the subgoal server
tmux select-pane -t 0
tmux send-keys "conda activate hi_learn" Enter
tmux send-keys "cd /home/noam/LLLwL/lcbc/baselines/zero_shot && python gpt_closed_loop_server.py" Enter

# Attach to the tmux session
tmux -2 attach-session -t $session_name
