import click
import numpy as np
import cv2
from pathlib import Path


@click.command()
@click.option('--video', '-v', help='Path to video file', required=True)
@click.option('--tracker', '-t', help='Path to tracker output', required=True)
@click.option('--dataset', '-d', type=click.Choice(['dronecrowd', 'upcount']), help='Dataset to use', required=True)
def main(video, tracker, dataset):
    video = Path(video)
    tracker = Path(tracker)
    
    cap = cv2.VideoCapture(str(video))
    lenght = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    gt_labels_path = Path(f'./eval/gt/{dataset}/MOT17-test/{video.stem}/gt/gt.txt')
    pred_labels_path = Path(f'./outputs/preds/{dataset}/MOT17-test/{tracker}/data/{video.stem}.txt')
    
    with open(gt_labels_path, 'r') as f:
        gt_labels = f.readlines()
        
    with open(pred_labels_path, 'r') as f:
        pred_labels = f.readlines()
        
    gt_labels = [l.split(',') for l in gt_labels]
    pred_labels = [l.split(',') for l in pred_labels]
    
    gt_labels = np.array(gt_labels).reshape(-1, 10)[:, :6].astype(np.int32)
    pred_labels = np.array(pred_labels).reshape(-1, 10)[:, :6].astype(np.int32)
    
    for frame_id in range(lenght):
        ret, frame = cap.read()
        
        lab_row = gt_labels[gt_labels[:, 0] == frame_id+1]
        pred_row = pred_labels[pred_labels[:, 0] == frame_id+1]
        
        for row in lab_row:
            topx, topy, w, h = row[2:6]
            cv2.rectangle(frame, (topx, topy), (topx+w, topy+h), (0, 255, 0), 2)
            
        for row in pred_row:
            topx, topy, w, h = row[2:6]
            cv2.rectangle(frame, (topx, topy), (topx+w, topy+h), (0, 0, 255), 2)
            
            
        cv2.imshow('frame', cv2.resize(frame, (1920, 1080)))
        k = cv2.waitKey(0)
        if k == 27:
            break
        
    cap.release()
    
if __name__ == '__main__':
    main()
        
