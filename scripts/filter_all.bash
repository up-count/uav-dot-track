#!/bin/bash

# parameters:
# dataset: upcount or dronecrowd
# use --track-cmc-flow to enable CMC flow tracking

if [ "$1" == "upcount" ]; then
    dataset="upcount"
elif [ "$1" == "dronecrowd" ]; then
    dataset="dronecrowd"
else
    echo "Usage: $0 upcount|dronecrowd"
    exit 1
fi

traj_dir="./outputs/preds/$dataset/MOT17-test/*"


for traj_file in $traj_dir; do
    if [ -d "$traj_file" ]; then
        python3 scripts/filter_trackings.py -i $traj_file
    fi
done
