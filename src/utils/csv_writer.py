from pathlib import Path

from src.tracking.track import Track


class CSVWriter:
    def __init__(self, output_dir, file_name):
        self.output_dir = output_dir
        self.file_name = file_name
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.file = open(f'{output_dir}/{file_name}.txt', 'w')
        
    def update(self, frame_id: int, track: Track):
        x, y, r = track.xyr

        #  for evaluation, the top left corner of the bounding box, size is always 20x20
        xtop, ytop = x - 10, y - 10
        w, h = 20, 20
        
        # f.write(f'{i+1},{track_id+1},{topx},{topy},{w},{h},1,-1,-1,-1\n')
        self.file.write(f'{frame_id+1},{track.track_id},{xtop},{ytop},{w},{h},{track.confidence},-1,-1,-1\n')
        
    def release(self):
        self.file.close()
        
    def __del__(self):
        self.release()
