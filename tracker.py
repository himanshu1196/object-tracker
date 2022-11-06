from filterpy.kalman import KalmanFilter
import numpy as np

class Tracker():
  count = 0

  def __init__(self) -> None:
    #define constant velocity model
    # dim_x is size of state vector (x, y, v_x, v_y) in this case
    # dim_z is size of measurement vector from the sensors (det_x, det_y) in this case
    self.kf = KalmanFilter(dim_x=4, dim_z=2)

    # Assign the initial value for the state (position and velocity)
    self.kf.x = np.array([500., 300., 0., 0.])

    # Define the state transition matrix:
    self.kf.F = np.array([[1., 0., 1., 0.],
                    [0., 1., 0., 1.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]])
    # Define the measurement function:
    self.kf.H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])

    # Define the covariance matrix. Here I take advantage of the fact that P already contains np.eye(dim_x), and just multiply by the uncertainty
    self.kf.P[2:,2:] *= 1000. #give high uncertainty to the unobservable initial velocities

    # Now assign the measurement noise.
    self.kf.R[:,:] *= 5.

    # assign the process noise. 
    self.kf.Q *= 0.01

    self.time_since_update = 0
    self.id = Tracker.count
    Tracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    #center of the bbox
    x = (bbox[0] + bbox[2])/2.
    y = (bbox[1] + bbox[3])/2.
    return np.array([x, y]).reshape((2, 1))
  
  def convert_state_to_rect(state):
    size = 10
    return np.array([state[0] - size/2, state[1] - size/2, state[0] + size/2, state[1] + size/2]).reshape((1,4))


  def update(self, det_box):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(Tracker.convert_bbox_to_z(det_box))

  def predict(self):
    """
    Advances the state vector and returns a rectangle around the predicted center.
    """
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(Tracker.convert_state_to_rect(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the rectangle around current center estimate.
    """
    return Tracker.convert_state_to_rect(self.kf.x)
