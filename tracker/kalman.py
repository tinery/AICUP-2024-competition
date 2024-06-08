from filterpy.kalman import KalmanFilter
import numpy as np
from enum import Enum
from collections import deque

class TrackStatus(Enum):
    Tentative = 0
    Confirmed = 1
    Coasted   = 2

class KalmanTracker(object):

    count = 1

    def __init__(self, y, R, wx, wy, vmax, w,h,F,center,tlbr,dt=1/30):
        
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        self.kf.R = R
        self.kf.P = np.zeros((4, 4))
        np.fill_diagonal(self.kf.P, np.array([1, vmax**2/3.0, 1,  vmax**2/3.0]))
        G = np.zeros((4, 2))
        G[0,0] = 0.5*dt*dt
        G[1,0] = dt
        G[2,1] = 0.5*dt*dt
        G[3,1] = dt
        Q0 = np.array([[wx, 0], [0, wy]])
        self.kf.Q = np.dot(np.dot(G, Q0), G.T)
        self.tlbr = None
        self.kf.x[0] = y[0]
        self.kf.x[1] = 0
        self.kf.x[2] = y[1]
        self.kf.x[3] = 0
        self.F = F
        self.smooth_feat = None
        history = 50
        self.features = deque([], maxlen=history)
        self.curr_feat = F
        self.alpha = 0.9
        self.his_vector = None
        self.center = None
        if F is not None:
            self.update_features(F)
        self.id = KalmanTracker.count
        KalmanTracker.count += 1
        self.age = 0
        self.death_count = 0
        self.birth_count = 0
        self.detidx = -1
        self.w = w
        self.h = h

        self.status = TrackStatus.Tentative
    def update_features(self, feat):
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update(self, y, R,tlbr):
        self.kf.update(y,R)
        self.tlbr = tlbr

    def predict(self):
        self.kf.predict()
        self.age += 1
        return np.dot(self.kf.H, self.kf.x)

    def get_state(self):
        return self.kf.x
    
    def distance(self, y, R):
        diff = y - np.dot(self.kf.H, self.kf.x)
        S = np.dot(self.kf.H, np.dot(self.kf.P,self.kf.H.T)) + R
        SI = np.linalg.inv(S)
        mahalanobis = np.dot(diff.T,np.dot(SI,diff))
        logdet = np.log(np.linalg.det(S))
        return mahalanobis[0,0] + logdet