from typing import List, Optional, Tuple

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from src.tracking.detection_result import DetectionResult
from src.tracking.sparse_optical_flow import SparseOpticalFlow
from src.tracking.track import Track, TrackParams
from src.tracking.track_state import TrackState
from src.tracking.classificator import Classificator


class Tracker:
    def __init__(self, dataset, max_age=5, min_hits=3, iou_threshold=0.4, use_flow = False, flow_scale_factor = 1.0, use_add_cls=False, use_pointflow=False):
        """ Initialize the tracker.
        
        Parameters
        ----------
        dataset : str
            The dataset name. One of ['dronecrowd', 'upcount']
        max_age : int
            The maximum number of frames a track can be inactive.
        min_hits : int
            The minimum number of hits to confirm a track.
        iou_threshold : float
            The threshold to consider a detection as a match.
        use_flow : bool
            Whether to use optical flow to update the tracks state in Kalman filter.
        flow_scale_factor : float
            The scale factor to downscale the optical flow image (reduce computation) if use_flow is True.
        use_add_cls : bool
            OUR IMPLEMENTATION: Whether to use the additional classification model to classify the object existence before confirming new tracks.
        use_pointflow : bool
            OUR IMPLEMENTATION: Whether to use the pointflow method to associate unmatched tracks and detections to make trajectories more stable.
        """
        self.iou_threshold = iou_threshold
        self.use_flow = use_flow
        self.flow_scale_factor = flow_scale_factor
        self.use_add_cls = use_add_cls
        self.use_pointflow = use_pointflow
        
        self.trackers = []
        self.tracks_counter = 0
        self.initialized = False
        self.prev_frame = None
        self.frame_shape = None
                
        self.default_track_params = TrackParams(
            max_age=max_age,
            min_hits=min_hits,
            use_add_cls=use_add_cls,
            use_pointflow=use_pointflow,
        )
        
        if self.use_flow:
            self.flow_estimator = SparseOpticalFlow(downscale_factor=flow_scale_factor)

        if self.use_add_cls:
            self.classificator = Classificator(f'./classification/best_model_epoch_{dataset}.pth')

        if self.use_pointflow:
            self.lk_params = dict( winSize  = (15, 15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))
            
            self.params = cv2.TrackerCSRT_Params()
            self.params.window_function = "hann"  # Hann window function
            self.params.padding = 3.0             # Increase padding for larger search area
            self.params.template_size = 200       # Set template size
            self.params.gsl_sigma = 1.0           # Gaussian spatial filter
            self.params.filter_lr = 0.02          # Learning rate
            self.params.num_hog_channels_used = 9 # Number of HOG channels
    
    def __call__(self, frame, frame_index, predictions):
        return self._update(frame, frame_index, predictions)
    
    def _add(self, predictions: List[DetectionResult], state=TrackState.NEW):
        for pred in predictions:
            self.tracks_counter += 1
            self.trackers.append(Track(self.tracks_counter, pred, state, self.default_track_params))
    
    def _update(self, frame: np.ndarray, frame_index: int, predictions: List[DetectionResult]):        
        if self.prev_frame is None:
            self.prev_frame = frame
            self.frame_shape = frame.shape
            self.default_track_params.frame_shape = frame.shape
        
        if not self.initialized and len(predictions) > 0:
            self._add(predictions, state=TrackState.CONFIRMED)
            self.initialized = True
            self.prev_frame = frame
            
            return [track for track in self.trackers if track.is_confirmed]

        if self.use_flow:
            H = self.flow_estimator.update(frame, frame_index)
        
        for track in self.trackers:
            track.step(warp=H if self.use_flow else None)
            
            if track.is_dead:
                self.trackers.remove(track)
            
        matched, unmatched_dets, unmatched_tracks = self._associate_detections_to_trackers(frame, predictions, warp=H if self.use_flow else None)
        
        for track_idx, pred in matched:
            t = self.trackers[track_idx]

            if self.use_add_cls:
                if t.is_new and t._active_counter >= (self.default_track_params.min_hits-3):
                    ret, prob = self.classificator.predict(frame, t.xyxy)

                    if ret:
                        t._objectness_list.append(prob)

            t.update(pred)
            
        for track_idx in unmatched_tracks:
            self.trackers[track_idx].mark_missed()
            
        for detection_idx in unmatched_dets:
            self._add([predictions[detection_idx]], state=TrackState.NEW)
        
        self.prev_frame = frame
        return [track for track in self.trackers if track.is_confirmed]
        
                
    def _associate_detections_to_trackers(self, frame, detections: List[DetectionResult], warp: Optional[np.ndarray] = None):
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []
        
        unmatched_detections = []
        unmatched_trackers = []
        matches = []
        
        if len(detections) > 0:
            detections_xyxys = np.array([d.xyxy for d in detections])
            trackers_xyxys = np.array([t.xyxy for t in self.trackers])
            
            iou_matrix = self._iou_batch(detections_xyxys, trackers_xyxys)
            
            if min(iou_matrix.shape) > 0:
                y, x = linear_sum_assignment(-iou_matrix)
                               
                matched_indices = np.array(list(zip(y, x)))
            else:
                matched_indices = np.empty(shape=(0,2))
                
            # handle unmatched detections
            for d in range(len(detections)):
                if d not in matched_indices[:, 0]:
                    unmatched_detections.append(d)
                    
            # handle unmatched trackers
            for t in range(len(self.trackers)):
                if t not in matched_indices[:, 1]:
                    unmatched_trackers.append(t)
                    
            # handle matches
            for m in matched_indices:
                if iou_matrix[m[0], m[1]] < self.iou_threshold:
                    unmatched_detections.append(m[0])
                    unmatched_trackers.append(m[1])
                else:
                    matches.append((m[1], detections[m[0]]))
        else:
            unmatched_trackers = list(range(len(self.trackers)))
        
        if self.use_pointflow and len(unmatched_trackers) > 0:
            matches_flow, unmatched_detections_flow, unmatched_trackers_flow = \
                self.associate_using_pointflow(frame, detections, matches, unmatched_detections, unmatched_trackers)

            if len(unmatched_trackers) < len(unmatched_trackers_flow):
                raise ValueError("The number of unmatched trackers should be not increased after using pointflow method.")
            
            if len(unmatched_detections_flow) < len(unmatched_detections):
                raise ValueError("The number of unmatched detections should be not increased after using pointflow method.")
            
            if len(matches_flow) > len(matches):
                raise ValueError("The number of matches should be not decreased after using pointflow method.")

            matches = matches_flow
            unmatched_detections = unmatched_detections_flow
            unmatched_trackers = unmatched_trackers_flow

        return matches, unmatched_detections, unmatched_trackers
    
    # https://github.com/RizwanMunawar/yolov7-object-tracking/blob/main/sort.py#L29C1-L44C14
    @staticmethod
    def _iou_batch(bb_test, bb_gt):        
        bb_gt = np.expand_dims(bb_gt, 0)
        bb_test = np.expand_dims(bb_test, 1)
        
        xx1 = np.maximum(bb_test[...,0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
        return(o)

    def associate_using_pointflow(self, frame: np.ndarray, detections: List[DetectionResult], matches: List[Tuple[int, DetectionResult]], unmatched_detections: List[int], unmatched_trackers: List[int]):
        raise NotImplementedError("This method should be implemented in the child class.")
        
        
    def predict_next_locations_using_pointflow(self, frame: np.ndarray, unmatched_trackers: List[int]):
        trackers_ids_to_update = np.array([i for i in unmatched_trackers if self.trackers[i].is_confirmed])

        if len(trackers_ids_to_update) == 0:
            print(f'[INFO] No trackers to update using pointflow.')
            return
        else:
            print(f'[INFO] Tracking {len(trackers_ids_to_update)} trackers using pointflow.')
        
        for tracker_id in trackers_ids_to_update:
            tracker = cv2.TrackerCSRT.create(self.params)

            _, _, w, h = self.trackers[tracker_id].xywh
            x1, y1, _, _ = self.trackers[tracker_id].xyxy

            bbox = (x1, y1, w, h)
            tracker.init(self.prev_frame, bbox)
            ret, bbox = tracker.update(frame)

            if ret:
                new_x1, new_y1, _, _ = bbox

                new_x = new_x1 + w // 2
                new_y = new_y1 + h // 2

                self.trackers[tracker_id]._pos_pointflow = DetectionResult(
                    label=self.trackers[tracker_id].label,
                    confidence=self.trackers[tracker_id].confidence,
                    x=new_x,
                    y=new_y,
                    w=w,
                    h=h,
                    frame_shape=self.frame_shape,
                    from_pointflow=True,
                )
            else:
                self.trackers[tracker_id]._pos_pointflow = None
                continue
