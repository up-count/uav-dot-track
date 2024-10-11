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

if [ "$dataset" == "upcount" ]; then
    video_dir="./datasets/upcount/test_videos/*.mp4"
elif [ "$dataset" == "dronecrowd" ]; then
    video_dir="./datasets/dronecrowd/test_videos/*.avi"
fi

name="base"
additionalArgs=""

if [ "$2" == "--track-cmc-flow" ] || [ "$3" == "--track-cmc-flow" ]; then
    additionalArgs=$additionalArgs" --track-cmc-flow"
    name=$name"+cmc"
fi

if [ "$2" == "--track-use-alt" ] || [ "$3" == "--track-use-alt" ]; then
    additionalArgs=$additionalArgs" --track-use-alt"
    name=$name"+alt"
fi


for video_file in $video_dir; do
    if [ -f "$video_file" ]; then
        python3 main.py --task pred --video $video_file --name $name --dataset $dataset --cache-det --device cuda $additionalArgs
    fi
done
