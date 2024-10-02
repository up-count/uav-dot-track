#!/bin/bash

# Directory containing video files
video_dir="./datasets/dronecrowd/test_videos/"

# Iterate over video files
for video_file in "$video_dir"/*.avi; do
    if [ -f "$video_file" ]; then
        python3 main.py --video $video_file --task none --name baseline --dataset dronecrowd --device cuda --cache-det
    fi
done
