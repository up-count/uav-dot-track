import cv2
import numpy as np

class VideoAltitudeReader:
    def __init__(self, video_path, use_alt):
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print(f'Failed to open video file {video_path}')
            raise FileNotFoundError

        self.resolution = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        self.file_name = video_path.split("/")[-1].split(".")[0]

        self.use_alt = use_alt

    def __iter__(self):
        return self
    
    def __next__(self):
        ret, frame = self.cap.read()
        
        if not ret:
            self.cap.release()
            raise StopIteration
        
        return frame
    
    def __del__(self):
        self.cap.release()

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
