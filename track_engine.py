import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from tracker import Tracker

def convert_bbox_to_x(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y] where x,y is the centre of the box
  """
  #center of the bbox
  x = (bbox[0] + bbox[2])/2.
  y = (bbox[1] + bbox[3])/2.
  return np.array([x, y, 0., 0.]).reshape((4, 1))


class TrackEngine():
  def __init__(self) -> None:
    self.tracks = []
    self.max_missed_frames = 5
    self.non_assignment_cost = 50

  def calculate_distance(self, detection, track):
    det_center = ((detection[0] + detection[2])/2., (detection[1] + detection[3])/2.)
    trk_s = track.get_state()[0]
    trk_center = ((trk_s[0] + trk_s[2])/2., (trk_s[1] + trk_s[3])/2.)
    distance = math.sqrt((trk_center[0] - det_center[0])**2 + (trk_center[1] - det_center[1])**2)
    return distance

  def calculate_cost_matrix(self, detections):
    n = len(self.tracks)
    m = len(detections)
    cost_mat = np.zeros(shape = (n, m))
    for i in range(n):
      for j in range(m):
        cost_mat[i,j] = self.calculate_distance(detections[j], self.tracks[i])

    return cost_mat

  def assign_detections_to_tracks(self, detections):
    n = len(self.tracks)
    m = len(detections)

    # Create tracks if no tracks vector found
    if (n == 0):
        for i in range(m):
            track = Tracker(convert_bbox_to_x(detections[i]))
            self.tracks.append(track)
            n += 1

    cost_mat = self.calculate_cost_matrix(detections)

    # Using Hungarian Algorithm assign the correct detected measurements
    # to predicted tracks
    assignment = []
    for _ in range(n):
        assignment.append(-1)
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    # print(row_ind, col_ind)
    for i in range(len(row_ind)):
        assignment[row_ind[i]] = col_ind[i]

    
    # Identify tracks with no assignment, if any
    un_assigned_tracks = []
    for i in range(len(assignment)):
      if (assignment[i] != -1):
        # check if cost is more than non_assignment_cost.
        # If cost is very high then un_assign (delete) the track
        if (cost_mat[i][assignment[i]] > self.non_assignment_cost):
          assignment[i] = -1
          un_assigned_tracks.append(i)
        pass
      else:
        un_assigned_tracks.append(i)
        pass
        # self.tracks[i].skipped_frames += 1

    # Now look for un_assigned detects
    un_assigned_detects = []
    for i in range(m):
      if i not in assignment:
        un_assigned_detects.append(i)

    return assignment, un_assigned_tracks, un_assigned_detects
  
  def update_tracks_for_detects(self, detections):
    assignment, un_assigned_tracks, un_assigned_detects = self.assign_detections_to_tracks(detections)

    # Update KalmanFilter state, lastResults and tracks trace
    for i in range(len(assignment)):
      if(assignment[i] != -1):
        self.tracks[i].update(detections[assignment[i]])
      else:
        #unassigned track
        pass
    
    # Start new tracks
    if(len(un_assigned_detects) != 0):
      for i in range(len(un_assigned_detects)):
        track = Tracker(convert_bbox_to_x(detections[un_assigned_detects[i]]))
        self.tracks.append(track)
    
    # clean up tracks which haven't been updated for a while
    del_tracks = []
    for i in range(len(self.tracks)):
      if (self.tracks[i].time_since_update > self.max_missed_frames):
        del_tracks.append(i)
    if len(del_tracks) > 0:
      for id in sorted(del_tracks, reverse = True):
        if id < len(self.tracks):
          del self.tracks[id]
          del assignment[id]
        else:
          print("ERROR: id is greater than length of tracks")

  def predict_all(self):
    predicted_states = []
    for i in range(len(self.tracks)):
      predicted_states.append(self.tracks[i].predict()[0])
    return predicted_states
