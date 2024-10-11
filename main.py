import cv2
import click
import yaml
import os
from tqdm import tqdm
import json

from src.tracking.tracker import Tracker
from src.tracking.detection_result import from_numpy_to_detection_results
from src.utils.csv_writer import CSVWriter
from src.utils.cache_helper import CacheHelper
from src.utils.video_reader import VideoAltitudeReader
from src.utils.video_writer import VideoWriter


@click.command()
@click.option('--name', type=str, help='Name of the experiment', default='exp')
@click.option('--dataset', type=click.Choice(['dronecrowd', 'upcount']), help='Dataset to use', required=True)
@click.option('--video', type=str, help='Path to the video file', required=True)
@click.option('--task', type=click.Choice(['vid', 'viz', 'pred', 'none', 'det']), help='Task to perform, can be multiple', required=True, multiple=True)
@click.option('--device', type=click.Choice(['cpu', 'cuda']), help='Device to run the model on', default='cpu')
@click.option('--cache-det', type=bool, is_flag=True, help='Cache detections')
@click.option('--cache-dir', type=str, help='Path to the cache directory', default='./cache')
@click.option('--track-max-age', type=int, help='Max age of the track', default=30)
@click.option('--track-min-hits', type=int, help='Min hits of the track', default=10)
@click.option('--track-iou-threshold', type=float, help='IOU threshold', default=0.2)
@click.option('--track-cmc-flow', type=bool, is_flag=True, help='Use optical flow')
@click.option('--track-use-alt', type=bool, is_flag=True, help='Use altitude information')
@click.option('--track-use-pointflow', type=bool, is_flag=True, help='Use pointflow for tracking')
def main(name, dataset, video, task, device, cache_det, cache_dir, track_max_age, track_min_hits, track_iou_threshold, track_cmc_flow, track_use_alt, track_use_pointflow):
    if not os.path.exists(video):
        print(f'Video file {video} does not exist')
        return
    
    if dataset == 'dronecrowd':
        dot_config = './uavdot/configs/dronecrowd.yaml'
        model = './checkpoints/dot_pd_dronecrowd_51.00.ckpt'

    elif dataset == 'upcount':
        dot_config = './uavdot/configs/upcount.yaml'
        model = './checkpoints/dot_pd_upcount_66.49.ckpt'
    else:
        print(f'Dataset {dataset} not supported')
        return
    
    if not os.path.exists(model):
        print(f'Model checkpoint {model} does not exist! Please download the model checkpoint available in the UAV-DOT repository and place it in the ./checkpoints/ directory.')
        return

    if not os.path.exists(dot_config):
        print(f'Dot config file {dot_config} does not exist. Please download submodules!')
        return
    
    with open(dot_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    video_source = VideoAltitudeReader(video, dataset=dataset, use_alt=track_use_alt)
        
    if 'vid' in task:
        write_video = VideoWriter('./outputs/videos/', video_source)
    
    if 'viz' in task:
        cv2.namedWindow('Tracking')
    
    if 'pred' in task:
        write_csv = CSVWriter(f'./outputs/preds/{dataset}/MOT17-test/{name}/data/', video_source.file_name)
        
    cache_helper = CacheHelper(cache_det, cache_dir + f'/{config["dataset"]}', video_source.file_name, video_source.resolution)
    cache_helper.set_model(model, device, config)
        
    ## Initialize the tracker
    tracker = Tracker(
        max_age=track_max_age,
        min_hits=track_min_hits,
        iou_threshold=track_iou_threshold,
        use_flow = track_cmc_flow,
        use_pointflow=track_use_pointflow,
        flow_scale_factor=2.0 if dataset == 'dronecrowd' else 4.0,
    )
    
    for i, (frame, alt) in enumerate(tqdm(video_source)):

        predictions = cache_helper(frame_id=i, image=frame)

        if 'det' in task:
            for pred in predictions:
                x, y, _ = pred
                
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        else:
            online_tracks = tracker(frame, i, from_numpy_to_detection_results(predictions, alt))

            for t in online_tracks:
                if 'vid' in task or 'viz' in task:
                    t.draw(frame, history_limit=100, pointsize=0.5 if dataset == 'dronecrowd' else 1.0)
                
                if 'pred' in task:
                    write_csv.update(frame_id=i, track=t)           
                    
        if 'vid' in task:
            write_video.update(frame)
        if 'viz' in task:
            cv2.imshow('Tracking', cv2.resize(frame, (1920, 1080)))
            key = cv2.waitKey(1)
            
            if key == 27:
                break
                        
    if 'vid' in task:
        write_video.release()
    
    if 'viz' in task:
        cv2.destroyAllWindows()
    
    if 'pred' in task:
        write_csv.release()

if __name__ == '__main__':
    main()
