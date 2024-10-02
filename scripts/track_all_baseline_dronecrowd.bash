#!/bin/bash

# Directory containing video files
video_dir="./datasets/dronecrowd/test_videos/*.avi"

for video_file in $video_dir; do
    if [ -f "$video_file" ]; then
        python3 main.py --task pred --video $video_file --name baseline --dataset dronecrowd --cache-det --device cuda --track-cmc-flow
    fi
done
