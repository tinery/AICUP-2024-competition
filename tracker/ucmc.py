
from __future__ import print_function
import numpy as np
from lap import lapjv
from scipy.spatial.distance import cosine
import math
from cython_bbox import bbox_overlaps as bbox_ious

from .kalman import KalmanTracker,TrackStatus

def is_point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2

def on_segment(p, q, r):
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def do_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    return False

def calculate_iou(box1, box2):
    top_intersection = max(box1[0], box2[0])
    left_intersection = max(box1[1], box2[1])
    bottom_intersection = min(box1[2], box2[2])
    right_intersection = min(box1[3], box2[3])
    
    intersection_width = max(0, right_intersection - left_intersection)
    intersection_height = max(0, bottom_intersection - top_intersection)
    
    intersection_area = intersection_width * intersection_height
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area != 0 else 0
    
    return iou

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious

def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def normalize_function(y):
    try:
        min_y = np.min(y)
        max_y = np.max(y)
        return 2 * (y - min_y) / (max_y - min_y)
    except:
        return y

def custom_function(x,i,limit_angle,p,slope = 8, tran =0.85):
    if np.prod(x) == -1:
        y = np.ones_like(x)
        y = np.where(p > 1.1,y*9999999,y)
        y = np.where(i == 0.5,y*0.1,y)
    else:
        x = np.where(i == 0.5,x*0.1,x)
        y = np.piecewise(x, 
                        [x <= limit_angle, x > limit_angle, x==-1], 
                        [lambda x: 1 + np.exp(slope * (tran*x - 1)), lambda x: 9999999,lambda x: 1]) 
        y = np.where(p > 1.1,9999999,y)
    return y

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

class UCMCTrack(object):
    def __init__(self,a1,a2,wx, wy,vmax, max_age, fps, dataset, high_score, use_cmc,detector = None):
        self.wx = wx
        self.wy = wy
        self.vmax = vmax
        self.dataset = dataset
        self.high_score = high_score
        self.max_age = max_age
        self.a1 = a1
        self.a2 = a2
        self.dt = 1.0/fps
        self.vector_error = 20
        self.use_cmc = use_cmc
        self.min_vector = 10
        self.trackers = []
        self.confirmed_idx = []
        self.coasted_idx = []
        self.tentative_idx = []
        self.detector = detector
        
    def update(self, dets,frame_id,flag_color,limit_vector,limit_angle,polygon,inverse_vector,camera,camera_ID):
        
        self.a1 = 100 if flag_color else 1.75
        self.a2 = 100 if flag_color else 1.75

        self.data_association(dets,frame_id,flag_color,limit_vector,limit_angle,polygon,inverse_vector,camera,camera_ID)
        
        self.associate_tentative(dets,flag_color,limit_vector,limit_angle,polygon,inverse_vector,camera,camera_ID)
        
        self.initial_tentative(dets)
        
        self.delete_old_trackers()
        
        self.update_status(dets)
    
    def data_association(self, dets,frame_id,flag_color,limit_vector,limit_angle,polygon,inverse_vector,camera,camera_ID):
        # Separate detections into high score and low score
        detidx_high = []
        detidx_low = []
        cam_list = ["1","2","3","4","5","6","7"]
        for i in range(len(dets)):
            if dets[i].conf >= self.high_score:
                detidx_high.append(i)
            else:
                detidx_low.append(i)

        # Predcit new locations of tracks
        for track in self.trackers:
            track.predict()
            if self.use_cmc:
                x,y = self.detector.cmc(track.kf.x[0,0],track.kf.x[2,0],track.w,track.h,frame_id)
                track.kf.x[0,0] = x
                track.kf.x[2,0] = y
        
        trackidx_remain = []
        self.detidx_remain = []

        ################## Associate high score detections with tracks ##################

        trackidx = self.confirmed_idx + self.coasted_idx
        num_det = len(detidx_high)
        num_trk = len(trackidx)

        for trk in self.trackers:
            trk.detidx = -1
        if num_det*num_trk > 0:
            cost_matrix = np.zeros((num_det, num_trk))
            cost_matrix_f = np.zeros((num_det, num_trk))
            cost_matrix_i = np.zeros((num_det, num_trk))
            cost_matrix_v = np.ones((num_det, num_trk))*-1
            cost_matrix_p = np.ones((num_det, num_trk))

            for i in range(num_det):
                det_idx = detidx_high[i]
                for j in range(num_trk):
                    trk_idx = trackidx[j]
                    cost_matrix[i,j] = self.trackers[trk_idx].distance(dets[det_idx].y, dets[det_idx].R) -1
                    cost_matrix_f[i,j] =  np.maximum(0.0, cosine(self.trackers[trk_idx].smooth_feat, dets[det_idx].feature)) 
                    if self.trackers[trk_idx].tlbr is not None:
                        cost_matrix_i[i,j] = calculate_iou(self.trackers[trk_idx].tlbr, dets[det_idx].tlbr)
                    if self.trackers[trk_idx].center is not None:
                        (x1_a, y1_a), (x2_a, y2_a) = self.trackers[trk_idx].center, dets[det_idx].center
                        cond = (self.trackers[trk_idx].h/2) if camera_ID in cam_list else 0
                        cond1 = (dets[det_idx].bb_height/2)if camera_ID in cam_list else 0
                        vector_a = np.array([x2_a - x1_a, y2_a - y1_a])
                        if (math.sqrt(vector_a[0]**2+vector_a[1]**2) > self.vector_error):
                            max_a = 0
                            for k in range (len(polygon)):
                                (x1_c, y1_c), (x2_c, y2_c) = inverse_vector[k][0], inverse_vector[k][1]
                                if is_point_in_polygon((x1_a, y1_a + cond),polygon[k]) or \
                                    is_point_in_polygon((x2_a, y2_a + cond1),polygon[k]):
                                    vector_a_ = np.array([x2_a - x1_a, y2_a - y1_a])
                                    vector_b_ = np.array([x2_c - x1_c, y2_c - y1_c])
                                    if is_point_in_polygon((x1_a, y1_a + cond ),polygon[k]) and \
                                    is_point_in_polygon((x2_a, y2_a + cond1),polygon[k]):
                                        max_a = cosine(vector_a_, vector_b_)
                                        max_a = 2 if max_a > 0.3 else max_a
                                    else:
                                        if cosine(vector_a_, vector_b_)  > max_a:
                                            max_a = cosine(vector_a_, vector_b_)
                            if max_a > 0:
                                cost_matrix_p[i,j] = max_a
                                
                        elif (math.sqrt(vector_a[0]**2+vector_a[1]**2) <= self.vector_error) and (math.sqrt(vector_a[0]**2+vector_a[1]**2) >  self.min_vector/2) :
                            for k in range (len(polygon)):
                                (x1_c, y1_c), (x2_c, y2_c) = inverse_vector[k][0], inverse_vector[k][1]
                                if (is_point_in_polygon((x1_a, y1_a + cond),polygon[k]) + is_point_in_polygon((x2_a, y2_a + cond1),polygon[k]))>0:
                                    vector_a_ = np.array([x2_a - x1_a, y2_a - y1_a])
                                    vector_b_ = np.array([x2_c - x1_c, y2_c - y1_c])
                                    if cosine(vector_a_, vector_b_) > 1.5:
                                        cost_matrix_p[i,j] = 2
                        if camera:
                            flag_intersect = False
                            for n in range(len(limit_vector)):
                                flag_intersect += do_intersect((x1_a, y1_a+ cond ),(x2_a, y2_a+ cond1),limit_vector[n][0],limit_vector[n][1])
                        else:
                            flag_intersect = do_intersect((x1_a, y1_a+ cond ),(x2_a, y2_a+ cond1),limit_vector[0],limit_vector[1])

                        if self.trackers[trk_idx].his_vector is not None:
                            (x1_b, y1_b), (x2_b, y2_b) = self.trackers[trk_idx].his_vector[0],self.trackers[trk_idx].his_vector[1]
                            vector_b = np.array([x2_b - x1_b, y2_b - y1_b])
                            if (math.sqrt(vector_a[0]**2+vector_a[1]**2) > self.vector_error) or (math.sqrt(vector_b[0]**2+vector_b[1]**2) > self.vector_error):
                                cost_matrix_v[i,j] = cosine(vector_a, vector_b) 
                        if flag_intersect:
                            cost_matrix_v[i,j]  = 2

            cost_matrix_i= np.where(cost_matrix_i>0.8,0.5,1)
            cost_matrix_v = custom_function(cost_matrix_v,cost_matrix_i,limit_angle,cost_matrix_p)

            if flag_color:
                cost_matrix = cost_matrix*cost_matrix_v 
            else:    
                cost_matrix = np.minimum(cost_matrix, cost_matrix_f)*cost_matrix_v
            matched_indices,unmatched_a,unmatched_b = linear_assignment(cost_matrix, self.a1)
            for i in unmatched_a:
                self.detidx_remain.append(detidx_high[i])
            for i in unmatched_b:
                trackidx_remain.append(trackidx[i])
            
            for i,j in matched_indices:
                det_idx = detidx_high[i]
                trk_idx = trackidx[j]
                dets[det_idx].his_box = self.trackers[trk_idx].tlbr
                self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R,dets[det_idx].tlbr)
                self.trackers[trk_idx].update_features(dets[det_idx].feature)
                self.trackers[trk_idx].death_count = 0
                self.trackers[trk_idx].detidx = det_idx
                self.trackers[trk_idx].status = TrackStatus.Confirmed
                dets[det_idx].track_id = self.trackers[trk_idx].id
                dets[det_idx].det_id = det_idx
                dets[det_idx].track_id_ = trk_idx
                if self.trackers[trk_idx].his_vector is not None:
                    dets[det_idx].his_pr = self.trackers[trk_idx].his_vector
                if self.trackers[trk_idx].center is not None:
                    (x1, y1), (x2, y2) = self.trackers[trk_idx].center, dets[det_idx].center
                    if (math.sqrt((x2-x1)**2+(y2-y1)**2) > self.min_vector):
                        self.trackers[trk_idx].his_vector = [self.trackers[trk_idx].center,dets[det_idx].center]
                    dets[det_idx].his = self.trackers[trk_idx].his_vector
                self.trackers[trk_idx].center = dets[det_idx].center

        else:
            self.detidx_remain = detidx_high
            trackidx_remain = trackidx

        ################# Associate low score detections with remain tracks ######################

        num_det = len(detidx_low)
        num_trk = len(trackidx_remain)
        if num_det*num_trk > 0:
            cost_matrix = np.zeros((num_det, num_trk))
            cost_matrix_f = np.zeros((num_det, num_trk))
            cost_matrix_i = np.zeros((num_det, num_trk))
            cost_matrix_v = np.ones((num_det, num_trk))*-1
            cost_matrix_p = np.ones((num_det, num_trk))
            for i in range(num_det):
                det_idx = detidx_low[i]
                for j in range(num_trk):
                    trk_idx = trackidx_remain[j]
                    cost_matrix[i,j] = self.trackers[trk_idx].distance(dets[det_idx].y, dets[det_idx].R) - 1
                    cost_matrix_f[i,j] =  np.maximum(0.0, cosine(self.trackers[trk_idx].smooth_feat, dets[det_idx].feature))
                    if self.trackers[trk_idx].tlbr is not None:
                        cost_matrix_i[i,j] = calculate_iou(self.trackers[trk_idx].tlbr, dets[det_idx].tlbr)
                    if self.trackers[trk_idx].center is not None:
                        (x1_a, y1_a), (x2_a, y2_a) = self.trackers[trk_idx].center, dets[det_idx].center
                        cond = (self.trackers[trk_idx].h/2) if camera_ID in ['5','6'] else 0
                        cond1 = (dets[det_idx].bb_height/2)if camera_ID in ['5','6'] else 0
                        vector_a = np.array([x2_a - x1_a, y2_a - y1_a])
                        if (math.sqrt(vector_a[0]**2+vector_a[1]**2) > self.vector_error):
                            max_a = 0
                            for k in range (len(polygon)):
                                (x1_c, y1_c), (x2_c, y2_c) = inverse_vector[k][0], inverse_vector[k][1]
                                if is_point_in_polygon((x1_a, y1_a + cond),polygon[k]) or \
                                    is_point_in_polygon((x2_a, y2_a + cond1),polygon[k]):
                                    vector_a_ = np.array([x2_a - x1_a, y2_a - y1_a])
                                    vector_b_ = np.array([x2_c - x1_c, y2_c - y1_c])
                                    if is_point_in_polygon((x1_a, y1_a + cond ),polygon[k]) and \
                                    is_point_in_polygon((x2_a, y2_a + cond1),polygon[k]):
                                        max_a = cosine(vector_a_, vector_b_)
                                        max_a = 2 if max_a > 0.3 else max_a
                                    else:
                                        if cosine(vector_a_, vector_b_)  > max_a:
                                            max_a = cosine(vector_a_, vector_b_)
                            if max_a > 0:
                                cost_matrix_p[i,j] = max_a

                        elif (math.sqrt(vector_a[0]**2+vector_a[1]**2) <= self.vector_error) and (math.sqrt(vector_a[0]**2+vector_a[1]**2) >  self.min_vector/2) :
                            for k in range (len(polygon)):
                                (x1_c, y1_c), (x2_c, y2_c) = inverse_vector[k][0], inverse_vector[k][1]
                                if (is_point_in_polygon((x1_a, y1_a + cond),polygon[k]) + is_point_in_polygon((x2_a, y2_a + cond1),polygon[k]))>0:
                                    vector_a_ = np.array([x2_a - x1_a, y2_a - y1_a])
                                    vector_b_ = np.array([x2_c - x1_c, y2_c - y1_c])
                                    if cosine(vector_a_, vector_b_) > 1.5:
                                        cost_matrix_p[i,j] = 2
                        if camera:
                            flag_intersect = False
                            for n in range(len(limit_vector)):
                                flag_intersect += do_intersect((x1_a, y1_a+ cond ),(x2_a, y2_a+ cond1),limit_vector[n][0],limit_vector[n][1])
                        else:
                            flag_intersect = do_intersect((x1_a, y1_a+ cond),(x2_a, y2_a+ cond1),limit_vector[0],limit_vector[1])
                        if self.trackers[trk_idx].his_vector is not None:
                            (x1_b, y1_b), (x2_b, y2_b) = self.trackers[trk_idx].his_vector[0],self.trackers[trk_idx].his_vector[1]
                            vector_b = np.array([x2_b - x1_b, y2_b - y1_b])
                            if (math.sqrt(vector_a[0]**2+vector_a[1]**2) > self.vector_error) or (math.sqrt(vector_b[0]**2+vector_b[1]**2) > self.vector_error):
                                cost_matrix_v[i,j] = cosine(vector_a, vector_b) 
                        if flag_intersect:
                            cost_matrix_v[i,j]  = 2
            cost_matrix_i= np.where(cost_matrix_i>0.8,0.5,1)
            cost_matrix_v = custom_function(cost_matrix_v,cost_matrix_i,limit_angle,cost_matrix_p)
            if flag_color:
                cost_matrix =cost_matrix*cost_matrix_v 
            else:    
                cost_matrix = np.minimum(cost_matrix, cost_matrix_f)*cost_matrix_v       
            matched_indices,unmatched_a,unmatched_b = linear_assignment(cost_matrix,self.a2)
            for i in unmatched_b:
                trk_idx = trackidx_remain[i]
                self.trackers[trk_idx].status = TrackStatus.Coasted
                # self.trackers[trk_idx].death_count += 1
                self.trackers[trk_idx].detidx = -1

            for i,j in matched_indices:
                det_idx = detidx_low[i]
                trk_idx = trackidx_remain[j]
                dets[det_idx].his_box = self.trackers[trk_idx].tlbr
                self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R,dets[det_idx].tlbr)
                self.trackers[trk_idx].update_features(dets[det_idx].feature)
                self.trackers[trk_idx].death_count = 0
                self.trackers[trk_idx].detidx = det_idx
                self.trackers[trk_idx].status = TrackStatus.Confirmed
                dets[det_idx].track_id = self.trackers[trk_idx].id
                dets[det_idx].det_id = det_idx
                dets[det_idx].track_id_ = trk_idx
                if self.trackers[trk_idx].his_vector is not None:
                    dets[det_idx].his_pr = self.trackers[trk_idx].his_vector
                if self.trackers[trk_idx].center is not None:
                    (x1, y1), (x2, y2) = self.trackers[trk_idx].center, dets[det_idx].center
                    if (math.sqrt((x2-x1)**2+(y2-y1)**2) > self.min_vector):
                        self.trackers[trk_idx].his_vector = [self.trackers[trk_idx].center,dets[det_idx].center]
                    dets[det_idx].his = self.trackers[trk_idx].his_vector
                self.trackers[trk_idx].center = dets[det_idx].center

    def associate_tentative(self, dets,flag_color,limit_vector,limit_angle,polygon,inverse_vector,camera,camera_ID):
        num_det = len(self.detidx_remain)
        num_trk = len(self.tentative_idx)
        cam_list = ["1","2","3","4","5","6","7"]
        cost_matrix = np.zeros((num_det, num_trk))
        cost_matrix_f = np.zeros((num_det, num_trk))
        cost_matrix_i = np.zeros((num_det, num_trk))
        cost_matrix_v = np.ones((num_det, num_trk))*-1
        cost_matrix_p = np.ones((num_det, num_trk))

        for i in range(num_det):
            det_idx = self.detidx_remain[i]
            for j in range(num_trk):
                trk_idx = self.tentative_idx[j]
                cost_matrix[i,j] = self.trackers[trk_idx].distance(dets[det_idx].y, dets[det_idx].R) - 1 
                cost_matrix_f[i,j] = np.maximum(0.0, cosine(self.trackers[trk_idx].smooth_feat, dets[det_idx].feature))  
                if self.trackers[trk_idx].tlbr is not None:
                    cost_matrix_i[i,j] = calculate_iou(self.trackers[trk_idx].tlbr, dets[det_idx].tlbr)
                if self.trackers[trk_idx].center is not None:
                        (x1_a, y1_a), (x2_a, y2_a) = self.trackers[trk_idx].center, dets[det_idx].center
                        cond = (self.trackers[trk_idx].h/2) if camera_ID in cam_list else 0
                        cond1 = (dets[det_idx].bb_height/2)if camera_ID in cam_list else 0
                        vector_a = np.array([x2_a - x1_a, y2_a - y1_a])
                        if (math.sqrt(vector_a[0]**2+vector_a[1]**2) > self.vector_error):
                            max_a = 0
                            for k in range (len(polygon)):
                                (x1_c, y1_c), (x2_c, y2_c) = inverse_vector[k][0], inverse_vector[k][1]
                                if is_point_in_polygon((x1_a, y1_a + cond),polygon[k]) or \
                                    is_point_in_polygon((x2_a, y2_a + cond1),polygon[k]):
                                    vector_a_ = np.array([x2_a - x1_a, y2_a - y1_a])
                                    vector_b_ = np.array([x2_c - x1_c, y2_c - y1_c])
                                    if is_point_in_polygon((x1_a, y1_a + cond ),polygon[k]) and \
                                    is_point_in_polygon((x2_a, y2_a + cond1),polygon[k]):
                                        max_a = cosine(vector_a_, vector_b_)
                                        max_a = 2 if max_a > 0.3 else max_a
                                    else:
                                        if cosine(vector_a_, vector_b_)  > max_a:
                                            max_a = cosine(vector_a_, vector_b_)
                            if max_a > 0:
                                cost_matrix_p[i,j] = max_a

                        elif (math.sqrt(vector_a[0]**2+vector_a[1]**2) <= self.vector_error) and (math.sqrt(vector_a[0]**2+vector_a[1]**2) >  self.min_vector/2) :
                            for k in range (len(polygon)):
                                (x1_c, y1_c), (x2_c, y2_c) = inverse_vector[k][0], inverse_vector[k][1]
                                if (is_point_in_polygon((x1_a, y1_a + cond),polygon[k]) + is_point_in_polygon((x2_a, y2_a + cond1),polygon[k]))>0:
                                    vector_a_ = np.array([x2_a - x1_a, y2_a - y1_a])
                                    vector_b_ = np.array([x2_c - x1_c, y2_c - y1_c])
                                    if cosine(vector_a_, vector_b_) > 1.5:
                                        cost_matrix_p[i,j] = 2
                        if camera:
                            flag_intersect = False
                            for n in range(len(limit_vector)):
                                flag_intersect += do_intersect((x1_a, y1_a+ cond ),(x2_a, y2_a+ cond1),limit_vector[n][0],limit_vector[n][1])
                        else:
                            flag_intersect = do_intersect((x1_a, y1_a+ cond ),(x2_a, y2_a+ cond1),limit_vector[0],limit_vector[1])
                        if self.trackers[trk_idx].his_vector is not None:
                            (x1_b, y1_b), (x2_b, y2_b) = self.trackers[trk_idx].his_vector[0],self.trackers[trk_idx].his_vector[1]
                            vector_b = np.array([x2_b - x1_b, y2_b - y1_b])
                            if (math.sqrt(vector_a[0]**2+vector_a[1]**2) > self.vector_error) or (math.sqrt(vector_b[0]**2+vector_b[1]**2) > self.vector_error):
                                cost_matrix_v[i,j] = cosine(vector_a, vector_b) 
                        if flag_intersect:
                            cost_matrix_v[i,j]  = 2

        cost_matrix_i= np.where(cost_matrix_i>0.8,0.5,1)
        cost_matrix_v = custom_function(cost_matrix_v,cost_matrix_i,limit_angle,cost_matrix_p)

        if flag_color:
                cost_matrix = cost_matrix*cost_matrix_v
        else:    
            cost_matrix = np.minimum(cost_matrix, cost_matrix_f)*cost_matrix_v
        matched_indices,unmatched_a,unmatched_b = linear_assignment(cost_matrix,self.a1)
        for i,j in matched_indices:
            det_idx = self.detidx_remain[i]
            trk_idx = self.tentative_idx[j]
            dets[det_idx].his_box = self.trackers[trk_idx].tlbr
            self.trackers[trk_idx].update(dets[det_idx].y, dets[det_idx].R,dets[det_idx].tlbr)
            self.trackers[trk_idx].update_features(dets[det_idx].feature)
            self.trackers[trk_idx].death_count = 0
            self.trackers[trk_idx].birth_count += 1
            self.trackers[trk_idx].detidx = det_idx
            dets[det_idx].track_id = self.trackers[trk_idx].id
            dets[det_idx].det_id = det_idx
            dets[det_idx].track_id_ = trk_idx
            if self.trackers[trk_idx].his_vector is not None:
                dets[det_idx].his_pr = self.trackers[trk_idx].his_vector
            if self.trackers[trk_idx].center is not None:
                (x1, y1), (x2, y2) = self.trackers[trk_idx].center, dets[det_idx].center
                if (math.sqrt((x2-x1)**2+(y2-y1)**2) > self.min_vector):
                    self.trackers[trk_idx].his_vector = [self.trackers[trk_idx].center,dets[det_idx].center]
                dets[det_idx].his = self.trackers[trk_idx].his_vector
            self.trackers[trk_idx].center = dets[det_idx].center
            if self.trackers[trk_idx].birth_count >= 2:
                self.trackers[trk_idx].birth_count = 0
                self.trackers[trk_idx].status = TrackStatus.Confirmed

        for i in unmatched_b:
            trk_idx = self.tentative_idx[i]
            self.trackers[trk_idx].death_count += 1
            self.trackers[trk_idx].detidx = -1

        unmatched_detidx = []
        for i in unmatched_a:
            unmatched_detidx.append(self.detidx_remain[i])
        self.detidx_remain = unmatched_detidx

    def initial_tentative(self,dets):
        for i in self.detidx_remain: 
            self.trackers.append(KalmanTracker(dets[i].y,dets[i].R,self.wx,self.wy,self.vmax, dets[i].bb_width,dets[i].bb_height,dets[i].feature,dets[i].center,dets[i].tlbr,self.dt))
            self.trackers[-1].status = TrackStatus.Tentative
            self.trackers[-1].detidx = i
            dets[i].track_id = self.trackers[-1].id
        self.detidx_remain = []

    def delete_old_trackers(self):
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            trk.death_count += 1
            i -= 1 
            if ( trk.status == TrackStatus.Coasted and trk.death_count >= self.max_age) or ( trk.status == TrackStatus.Tentative and trk.death_count >= 2):
                  self.trackers.pop(i)

    def update_status(self,dets):
        self.confirmed_idx = []
        self.coasted_idx = []
        self.tentative_idx = []
        for i in range(len(self.trackers)):
            detidx = self.trackers[i].detidx
            if detidx >= 0 and detidx < len(dets):
                self.trackers[i].h = dets[detidx].bb_height
                self.trackers[i].w = dets[detidx].bb_width
            if self.trackers[i].status == TrackStatus.Confirmed:
                self.confirmed_idx.append(i)
            elif self.trackers[i].status == TrackStatus.Coasted:
                self.coasted_idx.append(i)
            elif self.trackers[i].status == TrackStatus.Tentative:
                self.tentative_idx.append(i)

        
