import cv2
import numpy as np
from pathlib import Path

class VideoAltitudeReader:
    def __init__(self, video_path, dataset, use_alt):
        self.cap = cv2.VideoCapture(video_path)
        self.dataset = dataset
        
        if not self.cap.isOpened():
            print(f'[LOGS] Failed to open video file {video_path}')
            raise FileNotFoundError

        self.resolution = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        self.file_name = video_path.split("/")[-1].split(".")[0]

        self.use_alt = use_alt
        self.counter = 0

        # https://github.com/VisDrone/DroneCrowd/blob/master/STNNet/mytest.py#L126
        self.dronecrowd_attributes = {
            'small': [11, 15, 16, 22, 23, 24, 34, 35, 42, 43, 44, 61, 62, 63, 65, 69, 70, 74],
            'large': [17, 18, 75, 82, 88, 95, 101, 103, 105, 108, 111, 112]
        }

        if self.use_alt:
            if self.dataset == 'dronecrowd':
                # large means large objects, no large altitude
                val = 50 if int(self.file_name) in self.dronecrowd_attributes['large'] else 100

                self.gps = np.full((len(self), 4), val, dtype=np.float32)
            elif self.dataset == 'upcount':
                labels_path = video_path.replace('test_videos', 'test_gps').replace('mp4', 'txt')

                if not Path(labels_path).exists():
                    print(f'[LOGS] Failed to open labels file {labels_path}')
                    print('[LOGS] Disabling altitude')
                    self.use_alt = False
                else:
                    print(f'[LOGS] Using altitude from {labels_path}')
                    self.gps = np.loadtxt(labels_path, delimiter=',', skiprows=0, dtype=np.float32).reshape(-1, 4)
                    assert len(self.gps) == len(self)

    def __iter__(self):
        return self
    
    def __next__(self):
        ret, frame = self.cap.read()
        
        if not ret:
            self.cap.release()
            raise StopIteration
        
        if self.use_alt:
            alt = self.gps[self.counter][3]
        else:
            alt = -1

        self.counter += 1
        return frame, alt
    
    def __del__(self):
        self.cap.release()

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
