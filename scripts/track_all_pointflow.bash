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

name="our"
additionalArgs=""

if [ "$2" == "--track-use-add-cls" ] || [ "$3" == "--track-use-add-cls" ] || [ "$4" == "--track-use-add-cls" ]; then
    additionalArgs=$additionalArgs" --track-use-add-cls"
    name=$name"+cls"
fi

if [ "$2" == "--track-use-pointflow" ] || [ "$3" == "--track-use-pointflow" ] || [ "$4" == "--track-use-pointflow" ]; then
    additionalArgs=$additionalArgs" --track-use-pointflow"
    name=$name"+flow"
fi

if [ "$2" == "--track-use-cutoff" ] || [ "$3" == "--track-use-cutoff" ] || [ "$4" == "--track-use-cutoff" ]; then
    additionalArgs=$additionalArgs" --track-use-cutoff"
    name=$name"+cutoff"
fi


for video_file in $video_dir; do
    if [ -f "$video_file" ]; then
        echo "python3 main.py --task pred --video $video_file --name $name --dataset $dataset --cache-det --device cuda --track-use-alt --track-cmc-flow $additionalArgs"
        PYTHONWARNINGS="error" python3 main.py --task pred --video $video_file --name $name --dataset $dataset --cache-det --device cuda --track-use-alt --track-cmc-flow $additionalArgs
    fi
done
