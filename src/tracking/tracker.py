from typing import List, Optional, Tuple

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ThreadPoolExecutor

from src.tracking.detection_result import DetectionResult
from src.tracking.sparse_optical_flow import SparseOpticalFlow
from src.tracking.track import Track, TrackParams
from src.tracking.track_state import TrackState
from src.tracking.classificator import Classificator


class Tracker:
    def __init__(self, dataset, max_age=5, min_hits=3, use_flow = False, flow_scale_factor = 1.0, use_add_cls=False, use_pointflow=False, track_use_cutoff=False):
        """ Initialize the tracker.
        
        Parameters
        ----------
        dataset : str
            The dataset name. One of ['dronecrowd', 'upcount']
        max_age : int
            The maximum number of frames a track can be inactive.
        min_hits : int
            The minimum number of hits to confirm a track.
        use_flow : bool
            Whether to use optical flow to update the tracks state in Kalman filter.
        flow_scale_factor : float
            The scale factor to downscale the optical flow image (reduce computation) if use_flow is True.
        use_add_cls : bool
            OUR IMPLEMENTATION: Whether to use the additional classification model to classify the object existence before confirming new tracks.
        use_pointflow : bool
            OUR IMPLEMENTATION: Whether to use the pointflow method to associate unmatched tracks and detections to make trajectories more stable.
        track_use_cutoff : bool
            OUR IMPLEMENTATION: Whether to use the cut-off method to remove the track if the object is not detected for a long time based on the classification model.
        """
        self.use_flow = use_flow
        self.flow_scale_factor = flow_scale_factor
        self.use_add_cls = use_add_cls
        self.use_pointflow = use_pointflow
        self.track_use_cutoff = track_use_cutoff
        
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
            pointflow_method='csrt',
        )
        
        if self.use_flow:
            self.flow_estimator = SparseOpticalFlow(downscale_factor=flow_scale_factor)

        if self.use_add_cls or self.track_use_cutoff:
            self.classificator = Classificator(f'./classification/best_model_epoch_{dataset}.pth')

        if self.use_pointflow:
            self.lk_params = dict( winSize  = (15, 15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))
    
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
                continue

            if track._pointflow_counter % 10 == 0 and self.track_use_cutoff:
                ret, prob = self.classificator.predict(frame, track.xyr)

                track._pointflow_probs.append(prob)
                    
                if len(track._pointflow_probs) >= 3:
                    if np.mean(track._pointflow_probs) < 0.2:
                        self.trackers.remove(track)
                    else:
                        track._pointflow_probs = track._pointflow_probs[1:]
                continue
            
        matched, unmatched_dets, unmatched_tracks = self._associate_detections_to_trackers(frame, predictions, warp=H if self.use_flow else None)
        
        for track_idx, pred in matched:
            t = self.trackers[track_idx]

            if self.use_add_cls:
                if t.is_new and t._active_counter >= (self.default_track_params.min_hits-3):
                    ret, prob = self.classificator.predict(frame, t.xyr)

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
            detections_xyrs = np.array([d.xyr for d in detections])
            trackers_xyrs = np.array([t.xyr for t in self.trackers])
            
            dist_matrix = self._dist_batch(detections_xyrs, trackers_xyrs)
            
            if min(dist_matrix.shape) > 0.0:
                y, x = linear_sum_assignment(-dist_matrix)
                               
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
                if dist_matrix[m[0], m[1]] > 0:
                    matches.append((m[1], detections[m[0]]))
                else:
                    unmatched_detections.append(m[0])
                    unmatched_trackers.append(m[1])
                    
        else:
            unmatched_trackers = list(range(len(self.trackers)))
        
        if self.use_pointflow and len(unmatched_trackers) > 0:
            matches, unmatched_detections, unmatched_trackers = \
                self.associate_using_pointflow(frame, detections, matches, unmatched_detections, unmatched_trackers)

        return matches, unmatched_detections, unmatched_trackers
    
    @staticmethod
    def _dist_batch(xyr_test, xyr_gt):
        xy_test, maxdist_test = xyr_test[:, :2], xyr_test[:, 2:]
        xy_gt, maxdist_gt = xyr_gt[:, :2], xyr_gt[:, 2:]

        xy_dist = np.linalg.norm(xy_test[:, None] - xy_gt, axis=-1)
        
        maxdist_test = np.expand_dims(maxdist_test, 1)
        maxdist_gt = np.expand_dims(maxdist_gt, 0)

        maxdist_dist = np.minimum(maxdist_test, maxdist_gt)[:,:, 0]
        maxdist_dist *= 2 # diameter
        
        xy_dist[xy_dist > maxdist_dist] = maxdist_dist[xy_dist > maxdist_dist]

        xy_dist = 1 - xy_dist / maxdist_dist
       
        return xy_dist

    def associate_using_pointflow(self, frame: np.ndarray, detections: List[DetectionResult], matches: List[Tuple[int, DetectionResult]], unmatched_detections: List[int], unmatched_trackers: List[int]):
        self.estimate_next_locations(frame, unmatched_trackers)

        if len(unmatched_detections) > 0:
            matches, unmatched_detections, unmatched_trackers = self.associate_detections_using_pointflow(matches, detections, unmatched_detections, unmatched_trackers)

        if len(unmatched_trackers) > 0:
            matches, unmatched_trackers = self.associate_trackers_using_pointflow(matches, unmatched_trackers)

        return matches, unmatched_detections, unmatched_trackers
        
    def estimate_next_locations(self, frame: np.ndarray, unmatched_trackers: List[int]):
        params_list = [(self.trackers[t], self.prev_frame, frame) for t in unmatched_trackers if self.trackers[t].is_confirmed]

        with ThreadPoolExecutor(max_workers=12) as executor:
            # Map the computation to the thread pool
            futures = [executor.submit(lambda p: p[0].estimate_step(p[1], p[2]), params) for params in params_list]

            # Collect the results
            for future in futures:
                future.result()

    def associate_detections_using_pointflow(self, matches: List[Tuple[int, DetectionResult]], detections: List[DetectionResult], unmatched_detections: List[int], unmatched_trackers: List[int]):
        confirmed_unmatched_trackers = [t for t in unmatched_trackers if self.trackers[t].is_confirmed]

        if len(confirmed_unmatched_trackers) == 0:
            return matches, unmatched_detections, unmatched_trackers
        
        detections_xyrs = np.array([detections[d].xyr for d in unmatched_detections])
        pointflow_xyrs = np.array([self.trackers[t].xyr_pointflow for t in confirmed_unmatched_trackers])

        dist_matrix = self._dist_batch(detections_xyrs, pointflow_xyrs)

        if min(dist_matrix.shape) > 0:
            y, x = linear_sum_assignment(-dist_matrix)  
            matched_indices = np.array(list(zip(y, x)))
        else:
            matched_indices = np.empty(shape=(0,2))

        detections_to_remove = []
        trackers_to_remove = []

        for m in matched_indices:
            if dist_matrix[m[0], m[1]] > 0.0:
                tr_idx = confirmed_unmatched_trackers[m[1]]

                det = detections[unmatched_detections[m[0]]]
                det.from_pointflow = True

                matches.append((tr_idx, det))
                detections_to_remove.append(unmatched_detections[m[0]])
                trackers_to_remove.append(tr_idx)

        unmatched_detections = [d for i, d in enumerate(unmatched_detections) if i not in detections_to_remove]
        unmatched_trackers = [t for i, t in enumerate(unmatched_trackers) if i not in trackers_to_remove]

        return matches, unmatched_detections, unmatched_trackers


    def associate_trackers_using_pointflow(self, matches: List[Tuple[int, DetectionResult]], unmatched_trackers: List[int]):
        confirmed_unmatched_trackers = [t for t in unmatched_trackers if self.trackers[t].is_confirmed]

        if len(confirmed_unmatched_trackers) == 0:
            return matches, unmatched_trackers
        
        trackers_xyrs = np.array([self.trackers[t].xyr for t in confirmed_unmatched_trackers])
        pointflow_xyrs = np.array([self.trackers[t].xyr_pointflow for t in confirmed_unmatched_trackers])

        dist_matrix = self._dist_batch(pointflow_xyrs, trackers_xyrs)

        if min(dist_matrix.shape) > 0:
            y, x = linear_sum_assignment(-dist_matrix)  
            matched_indices = np.array(list(zip(y, x)))
        else:
            matched_indices = np.empty(shape=(0,2))

        trackers_to_remove = []

        for m in matched_indices:
            if dist_matrix[m[0], m[1]] > 0.0:
                tr_idx = confirmed_unmatched_trackers[m[0]]
                tr = self.trackers[tr_idx]

                det = DetectionResult(
                    x=tr.xyr_pointflow[0],
                    y=tr.xyr_pointflow[1],
                    r=tr.xyr_pointflow[2],
                    label=tr.label,
                    confidence=tr.confidence,
                    frame_shape=self.frame_shape,
                    from_pointflow=True,
                )

                matches.append((tr_idx, det))
                trackers_to_remove.append(tr_idx)

        unmatched_trackers = [t for t in unmatched_trackers if t not in trackers_to_remove]

        return matches, unmatched_trackers
