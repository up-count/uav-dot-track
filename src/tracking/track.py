from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import random
import cv2

from src.tracking.track_state import TrackState
from src.tracking.kalman_box_tracker import KalmanBoxTracker
from src.tracking.detection_result import DetectionResult
from src.tracking.track_state import TrackState

   
@dataclass
class TrackParams:
    max_age: int = 5
    min_hits: int = 3
    
    
class Track:
    def __init__(self, track_id: int, pred: DetectionResult, state: TrackState, track_params: TrackParams) -> None:
        self._track_id = track_id
        self._pred = [pred]
        self._state = state
        self._track_params = track_params
        
        self._active_counter = 1
        self._missing_counter = 0
        
        self._kbt = KalmanBoxTracker(pred)
        self._pos = pred
        self._pos_pointflow = None
        
        self._color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    def step(self, warp: Optional[np.ndarray] = None):
        self._pos = self._kbt.predict(warp)
        self._limit_pred_history()
 
    
    def update(self, pred: DetectionResult):
        self._pred.append(pred)
        self._active_counter += 1
        self._missing_counter = 0
        
        self._kbt.update(pred)
                        
        if self._active_counter > self._track_params.min_hits:
            self._state = TrackState.CONFIRMED
        
    def mark_missed(self):
        self._state = TrackState.MISSING
        self._missing_counter += 1
        self._active_counter = 0
        
        if self._missing_counter > self._track_params.max_age:
            self._state = TrackState.DEAD

    @property
    def state(self):
        return self._state

    @property
    def is_confirmed(self):
        return self._state.is_confirmed()
    
    @property
    def is_dead(self):
        return self._state.is_dead()

    @property
    def xywh(self):
        x, y, w, h = self._pos.xywh
        return int(x), int(y), int(w), int(h)
    
    @property
    def xyxy(self):
        x1, y1, x2, y2 = self._pos.xyxy
        return int(x1), int(y1), int(x2), int(y2)
    
    @property
    def xywh_pointflow(self):
        if self._pos_pointflow is None:
            return int(0), int(0), int(0), int(0)

        x, y, w, h = self._pos_pointflow.xywh
        return int(x), int(y), int(w), int(h)
    
    @property
    def xyxy_pointflow(self):
        if self._pos_pointflow is None:
            return int(0), int(0), int(0), int(0)

        x1, y1, x2, y2 = self._pos_pointflow.xyxy
        return int(x1), int(y1), int(x2), int(y2)

    @property
    def confidence(self):
        return self._pred[-1].confidence
    
    @property
    def color(self):
        return self._color
    
    @property
    def track_id(self):
        return self._track_id

    @property
    def label(self):
        return self._pred[-1].label

    def _limit_pred_history(self):
        self._pred = self._pred[-50:]

    def draw(self, frame, history_limit=100, pointsize=1.0):
        if self.is_confirmed:

            # if history_limit > 0:
            #     for i in range(1, len(self._pred)):
            #         x, y, _, _ = self._pred[i-1].xywh
            #         cv2.circle(frame, (int(x), int(y)), int(4*pointsize), self.color, -1)
 
            # x, y, _, _ = self.xywh
            # cv2.circle(frame, (int(x), int(y)), int(8*pointsize), self.color, -1)

            x1, y1, x2, y2 = self.xyxy

            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 4)
        
        return frame
