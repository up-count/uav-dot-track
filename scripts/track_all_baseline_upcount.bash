# Directory containing video files
video_dir="./datasets/upcount/test_videos/*.mp4"

for video_file in $video_dir; do
    if [ -f "$video_file" ]; then
        python3 main.py --task pred --video $video_file --name baseline --dataset upcount --cache-det --device cuda --track-cmc-flow
    fi
done
