import cv2
from pathlib import Path


class VideoWriter:
    def __init__(self, output_dir, file_name, cap):
        self.output_dir = output_dir
        self.file_name = file_name
        
        self.video_writer = None
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self.video_writer = cv2.VideoWriter(
            f'{self.output_dir}/{self.file_name}.avi',
            cv2.VideoWriter_fourcc(*'XVID'),
            float(cap.get(cv2.CAP_PROP_FPS)),
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            )

    def update(self, frame):     
        self.video_writer.write(frame)

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()
