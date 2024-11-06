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
    pointflow_method: str = ''

class PointFlowWrapper:
    def __init__(self, method: str, pred: DetectionResult):
        assert method in ['csrt']

        self._method = method
        self._status = None
        self._initialized = False

        if method == 'csrt':
            self.params = cv2.TrackerCSRT_Params()
            self.params.use_gray = True
            self.params.template_size = int(4*pred.xyr[2])

            self.tracker = cv2.TrackerCSRT.create(self.params)

    def update(self, prev_frame: np.ndarray, frame: np.ndarray, track: 'Track'):
        x, y, r = track.xyr
        
        if not self._initialized:
            bbox = (x-r, y-r, 2*r, 2*r)
            self.tracker.init(prev_frame, bbox)
            self._initialized = True

        ret, bbox = self.tracker.update(frame)

        if ret:
            new_x1, new_y1, new_w, new_h = bbox

            new_xc = new_x1 + new_w // 2
            new_yc = new_y1 + new_h // 2

            self._status = DetectionResult(
                label=track.label,
                confidence=track.confidence,
                x=new_xc,
                y=new_yc,
                r=r,
                frame_shape=track._pred[-1].frame_shape,
                from_pointflow=True,
            )
        else:
            self._status = None
                

    def xyr(self):
        if self._status is None:
            return 0, 0, 1
        else:
            return self._status.xyr

class Track:
    def __init__(self, track_id: int, pred: DetectionResult, state: TrackState, track_params: TrackParams) -> None:
        self._track_id = track_id
        self._pred = [pred]
        self._state = state
        self._prev_state = state
        self._track_params = track_params
        
        self._missing_counter = 0
        self._pointflow_counter = 0
        self._pointflow_probs = []
        self._pointflow = None
        
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
 
    def estimate_step(self, prev_frame: np.ndarray, frame: np.ndarray):
        if self._pointflow is None:
            self._pointflow = PointFlowWrapper(self._track_params.pointflow_method, self._pred[-1])

        self._pointflow.update(prev_frame, frame, self)

    def update(self, pred: DetectionResult):
        self._pred.append(pred)
        self._missing_counter = 0
        self._active_counter += 1
        
        self._kbt.update(pred)

        if not pred.from_pointflow:
            self._pointflow = None
            self._pointflow_counter = 0
        else:
            self._pointflow_counter += 1

        if self.is_new and self._active_counter >= self._track_params.min_hits:
            if self._track_params.use_add_cls:
                if len(self._objectness_list) > 1:
                    conf = np.mean(self._objectness_list)
                else:
                    conf = 0.0
            else:
                conf = 1.0

            if conf >= 0.5:
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
        self._active_counter = 0
        
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
    def xyr(self):
        x, y, r = self._pos.xyr
        return int(x), int(y), int(r)
    
    @property
    def xyr_pointflow(self):
        if self._pointflow is None:
            return int(0), int(0), int(1)
        
        x, y, r = self._pointflow.xyr()
        return int(x), int(y), int(r)
    
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
        x, y, r = self.xyr

        frame_height, frame_width = self._track_params.frame_shape[:2]

        if r < 5 or r < 5:
            self._prev_state = self._state
            self._state = TrackState.DEAD
            return
        
        if x-r < 0 or y-r < 0:
            self._prev_state = self._state
            self._state = TrackState.DEAD
            return
                
        if x+r >= frame_width or y+r >= frame_height:
            self._prev_state = self._state
            self._state = TrackState.DEAD
            return

    def draw(self, frame, history_limit=100, pointsize=1.0):
        if self.is_confirmed:
            if history_limit > 0:
                for i in range(1, len(self._pred)):
                    x, y, _ = self._pred[i-1].xyr

                    cv2.circle(frame, (int(x), int(y)), int(4*pointsize), self.color, -1)
 
            x, y, _ = self.xyr
            cv2.circle(frame, (int(x), int(y)), int(8*pointsize), self.color, -1)
        
        return frame
