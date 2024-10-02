import cv2
from pathlib import Path

from src.utils.video_reader import VideoAltitudeReader


class VideoWriter:
    def __init__(self, output_dir, video_source: VideoAltitudeReader):
        self.output_dir = output_dir
        self.file_name = video_source.file_name
        
        self.video_writer = None
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.video_writer = cv2.VideoWriter(
            f'{self.output_dir}/{self.file_name}.avi',
            cv2.VideoWriter_fourcc(*'XVID'),
            video_source.fps,
            video_source.resolution,
            )

    def update(self, frame):     
        self.video_writer.write(frame)

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()
