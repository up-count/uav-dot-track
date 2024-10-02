from typing import Tuple

import numpy as np
from pathlib import Path
import os
import sys
import torch
from enum import Enum

from src.utils.image import preprocess_image

# add uavdot to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../uavdot'))
from src.model.dot_regressor import DotRegressor

class CacheOptions(Enum):
    NO_USE = 0
    CREATE = 1
    USE = 2


class CacheHelper:
    def __init__(self, cache_det: bool, cache_dir: str, file_name: str, video_resolution: Tuple[int, int]) -> None:
        self.cache_det = cache_det
        self.cache_dir = Path(os.path.join(cache_dir, file_name))
        self.cache_does_not_exist = not self.cache_dir.exists()
        self.video_w, self.video_h = video_resolution
        self.model_w, self.model_h = None, None
        
        self.regressor = None
        self.config = None
        
        if not self.cache_det:
            self.status = CacheOptions.NO_USE
            print(f'[LOGS] Detection caching disabled')
        elif self.cache_does_not_exist:
            self.status = CacheOptions.CREATE
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f'[LOGS] Cache directory created at {self.cache_dir}')
        else:
            self.status = CacheOptions.USE
            print(f'[LOGS] Cache directory found at {self.cache_dir}')
        
    def set_model(self, model_path: str, engine: str, config: dict) -> None:
        if self.status == CacheOptions.CREATE or self.status == CacheOptions.NO_USE:
            print(f'[LOGS] Loading model')
            self.config = config
            self.model_w, self.model_h = self.config['image_size']
            
            self.regressor = DotRegressor.load_from_checkpoint(
                checkpoint_path=model_path,
                map_location=engine
            )
            
            self.regressor.eval()
            
            if engine == 'cuda':
                self.regressor.cuda()
        else:
            print(f'[LOGS] Skipping model loading')
            
    def infer_model(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.regressor is None:
            raise ValueError('CacheHelper not initialized properly, call .set_model() first')
        
        with torch.no_grad():
            input_tensor = preprocess_image(
                frame=image,
                image_size=self.config['image_size'],
                mean=self.config['data_mean'],
                std=self.config['data_std']
            )
            
            if self.regressor.device.type == 'cuda':
                input_tensor = input_tensor.cuda()
                
            dot_count = self.regressor.forward(input_tensor)[0]
            
            points = self.regressor.postprocessing(dot_count, thresh=0.2)[0]
            points[:, 3] = torch.sigmoid(points[:, 3])
            
            concat = points[:, 1:4].cpu().numpy()

            concat[:, 0] = concat[:, 0] * self.video_w / self.model_w
            concat[:, 1] = concat[:, 1] * self.video_h / self.model_h
            
            return concat

    def __call__(self, frame_id, image):
        if self.status == CacheOptions.NO_USE:
            return self.infer_model(image)
        elif self.status == CacheOptions.CREATE:
            concat = self.infer_model(image)
            np.save(self.cache_dir / f'{frame_id:05d}.npy', concat)
            
            return concat
        else:
            concat = np.load(self.cache_dir / f'{frame_id:05d}.npy')
            
            return concat
