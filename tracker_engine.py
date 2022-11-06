from tracker import Tracker
import numpy as np

class TrackEngine():
  def __init__(self) -> None:
    self.tracks = []

  def calculate_distance(self, detection, track):
    det_center = ((detection[0] + detection[2])/2., (detection[1] + detection[3])/2.)
    trk_s = track.get_state()
    trk_center = ((trk_s[0] + trk_s[2])/2., (trk_s[1] + trk_s[3])/2.)
    distance = (trk_center[0] - det_center[0])**2 + (trk_center[1] - det_center[1])**2
    return distance

  def calculate_cost_matrix(self, detections):
    cost_mat = np.zeros(shape = ())
    return

  def assign_detections_to_tracks(self, detections, non_assignment_cost):
    return
