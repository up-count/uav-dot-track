#!/bin/bash

# Directory containing video files
video_dir="./datasets/upcount-track/test_videos/"

# Iterate over video files
for video_file in "$video_dir"/*.mp4; do
    if [ -f "$video_file" ]; then
        python3 main.py --video $video_file --task none --name baseline --dataset upcount --device cuda --cache-det
    fi
done
