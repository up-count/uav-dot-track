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
    use_add_cls: bool = False
    use_pointflow: bool = False
    frame_shape: Tuple[int, int] = (0, 0)
    

class Track:
    def __init__(self, track_id: int, pred: DetectionResult, state: TrackState, track_params: TrackParams) -> None:
        self._track_id = track_id
        self._pred = [pred]
        self._state = state
        self._prev_state = state
        self._track_params = track_params
        
        self._missing_counter = 0
        self._pos_pointflow = None
        
        assert track_params.frame_shape != (0, 0), 'Frame shape is not set'

        self._kbt = KalmanBoxTracker(pred, track_params.frame_shape)
        self._pos = pred

        self._color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        if state.is_confirmed():
            self._active_counter = self._track_params.min_hits
            self._objectness_list = [1.0, 1.0, 1.0]
        else:
            self._active_counter = 1
            self._objectness_list = []

    def step(self, warp: Optional[np.ndarray] = None):
        self._pos = self._kbt.predict(warp)
        self._limit_pred_history()
        self._remove_if_out_of_frame()
 
    def update(self, pred: DetectionResult):
        self._pred.append(pred)
        self._missing_counter = 0
        self._active_counter += 1
        
        self._kbt.update(pred)

        if self.is_new and self._active_counter >= self._track_params.min_hits:
            if self._track_params.use_add_cls:
                if len(self._objectness_list) > 1:
                    conf = np.mean(self._objectness_list)
                else:
                    conf = 0.0
            else:
                conf = 1.0

            if conf >= 0.6:
                self._prev_state = TrackState.NEW
                self._state = TrackState.CONFIRMED
            else:
                self._prev_state = TrackState.NEW
                self._state = TrackState.DEAD
                        
        if self.is_missing:
            self._prev_state = TrackState.MISSING
            self._state = TrackState.CONFIRMED
        
    def mark_missed(self):
        if self.is_confirmed:
            self._prev_state = TrackState.CONFIRMED
            self._state = TrackState.MISSING

        self._missing_counter += 1
        
        if self._missing_counter > self._track_params.max_age:
            self._prev_state = TrackState.MISSING
            self._state = TrackState.DEAD

    @property
    def state(self):
        return self._state

    @property
    def is_new(self):
        return self._state.is_new()

    @property
    def is_confirmed(self):
        return self._state.is_confirmed()
    
    @property
    def is_missing(self):
        return self._state.is_missing()

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

    def _remove_if_out_of_frame(self):
        x, y, w, h = self.xywh
        x1, y1, x2, y2 = self.xyxy
        frame_height, frame_width = self._track_params.frame_shape[:2]

        if w < 5 or h < 5:
            self._prev_state = self._state
            self._state = TrackState.DEAD
            return
        
        if x1 < 0 or y1 < 0:
            self._prev_state = self._state
            self._state = TrackState.DEAD
            return
        
        if x2 < x1 or y2 < y1:
            self._prev_state = self._state
            self._state = TrackState.DEAD
            return
        
        if x2 >= frame_width or y2 >= frame_height:
            self._prev_state = self._state
            self._state = TrackState.DEAD
            return
        
        if x1 + w >= frame_width or y1 + h >= frame_height:
            self._prev_state = self._state
            self._state = TrackState.DEAD
            return

    def draw(self, frame, history_limit=100, pointsize=1.0):
        if self.is_confirmed:
            if self._pred[-1].from_pointflow:
                color = (192, 15, 252)
                pointsize *= 1.5
            else:
                color = self.color

            if history_limit > 0:
                for i in range(1, len(self._pred)):
                    x, y, _, _ = self._pred[i-1].xywh

                    cv2.circle(frame, (int(x), int(y)), int(4*pointsize), color, -1)
 
            x, y, _, _ = self.xywh
            cv2.circle(frame, (int(x), int(y)), int(8*pointsize), color, -1)
        
        return frame
