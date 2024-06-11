from ultralytics import YOLO
import os,cv2
import argparse
from skimage.metrics import structural_similarity as ssim
from fast_reid.fast_reid_interfece import FastReIDInterface
import os.path as osp
from tracker.ucmc import UCMCTrack
from detector.mapper import Mapper
import numpy as np

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
cam_ig = [(705,160),(750,0)], [(737,390),(0,220)],[(545,305),(500,0)],[(295,720),(335,282)],[(666,380),(785,0)],\
         [(835,204),(481,81)],[(400,303),(407,215)],[(381,717),(319,214)]
cam_angle = [0.2,0.2,0.4,0.15,0.2,0.3,0.3,0.2]
cam_contour = [[[(419,133),(677,146),(741,0),(603,0)],[(745,155),(1106,197),(854,0),(757,0)]],\
               [[(737,390),(0,683),(0,220)],[(737,390),(1070,254),(0,120),(0,220)]],\
               [[(0,320),(492,298),(484,96),(379,105)],[(588,292),(1038,287),(586,92),(517,95)]],\
               [[(436,715),(1278,713),(1275,458),(385,274),(321,304)],[(99,715),(0,714),(0,391),(248,279),(345,270)]],\
               [[(99,312),(615,326),(779,0),(721,0)],[(757,362),(1254,343),(838,0),(784,0)]],\
               [[(227,308),(779,182),(478,80),(111,104)],[(800,190),(483,78),(773,66),(1250,179)]],\
               [[(167.2,199.2),(403.2,196.0),(396.0,347.20),(0.0,364.624),(0.0,299.20)],[(404.8,194.4),(396.8,349.6),(1076,325),(732.0,188)],\
                [(64.0,165.6),(972.8,156.0),(998.4,184.7),(66.3,197.6)]],\
               [[(300,713),(282,230),(131,244),(0,327),(0,715)],[(427,713),(1277,709),(1267,480),(381,206),(323,217)]]]
cam_vector = [[[(650,70),(550,166)],[(850,165),(795,40)]],\
              [[(24,294),(538,462)],[(765,302),(191,201)]],\
              [[(360,173),(259,309)],[(743,286),(680,173)]],\
              [[(973,505),(518,344)],[(206,334),(0,480)]],\
              [[(615,137),(407,320)],[(974,362),(871,159)]],\
              [[(354,105),(609,244)],[(1125,216),(635,77)]],\
              [[(286.062, 210.2),(167.1,342.5)],[(699.4,309.44),(560.8,197.6)],[(636.8,180.0),(477.6,183.2)]],\
              [[(154,311),(90,441)],[(698,407),(519,318)]]]

def IQM_FFT(img_gray):
    rows, cols = img_gray.shape
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)    
    m = np.max(np.abs(fshift))
    sum = np.sum(f > (m/1000))
    IQM = sum/(rows*cols )
    return IQM

def IQM_FFT_ignore(img_gray_ignore,areas):
    rows, cols = img_gray_ignore.shape
    f = np.fft.fft2(img_gray_ignore)
    fshift = np.fft.fftshift(f)    
    m = np.max(np.abs(fshift))
    sum = np.sum(f > (m/1000))
    IQM = sum/(rows*cols -areas )
    return IQM

def colorless(image):
    flag = False
    colors = ("r", "g", "b")
    add_margin = 10
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hists = []
    max_hist = []
    for i in range(len(colors)):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        max_ = np.argmax(hist)
        max_hist.append(max_)
        hists.append(hist)
    sorted_indices = np.argsort(max_hist)
    min_o = sorted_indices[0]
    mid_o = sorted_indices[1]
    max_o = sorted_indices[2]
    offset_1 = max_hist[mid_o] - max_hist[min_o]
    offset_2 = max_hist[max_o] - max_hist[min_o]
    offset_3 = max_hist[max_o] - max_hist[mid_o]
    mid_arr = hists[mid_o][offset_1+add_margin:-(offset_3+add_margin)]
    min_arr = hists[min_o][add_margin:-(offset_2+add_margin)]
    max_arr = hists[max_o][offset_2+add_margin:-add_margin]
    score = 2*mid_arr - min_arr - max_arr
    if (score.mean()) == 0:
        flag = True
    return flag

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def list_folders_in_directory(parent_directory):
    try:  
        items = os.listdir(parent_directory)
        folders = [item for item in items if os.path.isdir(os.path.join(parent_directory, item))]
        return folders
    except Exception as e:
        print(f"error: {e}")
        return []

def are_images_similar(image1, image2, threshold=0.99):
    if image1.shape != image2.shape:
        return False
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    (score, _) = ssim(gray1, gray2, full=True)
    print(score)
    return score >= threshold

class Detection:

    def __init__(self, id, bb_left = 0, bb_top = 0, bb_width = 0, bb_height = 0, conf = 0, det_class = 0):
        self.id = id
        self.bb_left = bb_left
        self.bb_top = bb_top
        self.bb_width = bb_width
        self.bb_height = bb_height
        self.conf = conf
        self.det_class = det_class
        self.track_id = 0
        self.his = None
        self.his_box = None
        self.his_pr = None
        self.tlbr = None
        self.track_id_ = 0
        self.det_id = 0
        self.y = np.zeros((2, 1))
        self.R = np.eye(4)
        self.feature = None
        self.center = None

    def __str__(self):
        return 'd{}, bb_box:[{},{},{},{}], conf={:.2f}, class{}, uv:[{:.0f},{:.0f}], mapped to:[{:.1f},{:.1f}]'.format(
            self.id, self.bb_left, self.bb_top, self.bb_width, self.bb_height, self.conf, self.det_class,
            self.bb_left+self.bb_width/2,self.bb_top+self.bb_height,self.y[0,0],self.y[1,0])

    def __repr__(self):
        return self.__str__()

class Detector:
    def __init__(self,args):
        self.seq_length = 0
        self.gmc = None
        fast_reid_config = args.model_config
        fast_reid_weights = args.w_reid
        device = "gpu"
        self.encoder = FastReIDInterface(fast_reid_config, fast_reid_weights, device)
        self.model = YOLO(args.w_detection)
    def load(self,cam_para_file):
        self.mapper = Mapper(cam_para_file,"MOT17")
        
    def get_dets(self, img,conf_thresh = 0,det_classes = [0]):
        
        dets = []
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        results = self.model(frame,imgsz = 1088)
        features_keep = self.encoder.inference(img, results)
        features_keep /= np.linalg.norm(features_keep)
        det_id = 0
        for i, box in enumerate(results[0].boxes):
            feature = features_keep[i]
            conf = box.conf.cpu().numpy()[0]
            bbox = box.xyxy.cpu().numpy()[0]
            cls_id  = box.cls.cpu().numpy()[0]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= 10 and h <= 10 or cls_id not in det_classes or conf <= conf_thresh:
                continue
            det = Detection(det_id)
            det.bb_left = bbox[0]
            det.bb_top = bbox[1]
            det.bb_width = w
            det.bb_height = h
            det.conf = conf
            det.det_class = cls_id
            det.y,det.R = self.mapper.mapto([det.bb_left,det.bb_top,det.bb_width,det.bb_height])
            det_id += 1
            det.feature = feature
            det.center = np.array([bbox[0]+w/2, bbox[1]+h/2])
            det.tlbr = bbox
            dets.append(det)
        return dets
    
def main(args):

    class_list = [0]
    fps = 1
    detector = Detector(args)
    tracker = UCMCTrack(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score,False,None)
    path = args.path +args.name
    if osp.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    pr_frame = None
    flag_duplicate = False
    results = []
    save_dir = "./results/"    
    cam_prv = '0'

    for frame_id, img_path in enumerate(files, 1):
        camera = False
        camera_ID = img_path[-11]
        detector.load(args.cam_para+camera_ID+'.txt')
        flag_change_cam = False
        # if camera_ID != "0":
        #     continue
        if camera_ID != cam_prv:
            flag_change_cam = True
            cam_prv = camera_ID
        limit = cam_ig[int(camera_ID)]
        limit_angle = cam_angle[int(camera_ID)]
        frame_img = cv2.imread(img_path)
        polygon = cam_contour[int(camera_ID)]
        inverse_vector = cam_vector[int(camera_ID)]
        if camera_ID == "5":
            if args.name[:4] == "0902" or args.name[:4] == "0903":
                camera = True
                detector.load(args.cam_para+camera_ID+'_1'+'.txt')
                limit = [[(835,204+247),(481,81+247)],[(114.2,200),(365.7,280.)]]
                polygon = [[(465.6,322.40),(120.8,341.6),(221.5,586.21),(924.0,481.6)],\
                        [(750.4,308.0),(472.0,323.2),(925.59,480.8),(1260.8,422.4)], \
                        [(111.9,200.79),(63.2,201.6),(72.8,292.8),(356.8,280.8)],\
                        [(112.8,199.2),(364.8,282.4),(589.59,275.20),(238.4,206.4)]]
                inverse_vector = [[(352.8,348.8),(475.0,500.0)],\
                                [(895.8,439.6),(693.599,341.6)],\
                                [(127.45,229.36),(175.0,267.5)],\
                                [(427.2,270.40),(347.2,245.6)],]
                
        flag_color = colorless(frame_img)
        # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # iqm_fg =IQM_FFT(img_gray)
        if pr_frame is not None:
            flag_duplicate = are_images_similar(frame_img,pr_frame)
        pr_frame = frame_img.copy()
        if flag_change_cam:
            tracker.__init__(args.a, args.a, args.wx, args.wy, args.vmax, args.cdt, fps, "MOT", args.high_score,False,None)
        if flag_duplicate == False:
            dets = detector.get_dets(frame_img,args.conf_thresh,class_list)
            tracker.update(dets,frame_id,flag_color,limit,limit_angle,polygon,inverse_vector,camera,camera_ID)
        for i in range(len(polygon)):
            cv2.polylines(frame_img, [np.array(polygon[i]).astype(int)], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.arrowedLine(frame_img, (int(inverse_vector[i][0][0]),int(inverse_vector[i][0][1])), \
                            (int(inverse_vector[i][1][0]),int(inverse_vector[i][1][1])), (255,0,255), 1, tipLength=0.2)
        if  camera:
            for i in range(len(limit)):
                print(limit[i][0][0])
                cv2.line(frame_img, (int(limit[i][0][0]),int(limit[i][0][1])), (int(limit[i][1][0]),int(limit[i][1][1])), (0,255,255), 2)
        else:
            cv2.line(frame_img, (int(limit[0][0]),int(limit[0][1])), (int(limit[1][0]),int(limit[1][1])), (0,255,255), 2)
        for i,det in enumerate(dets):
            cv2.rectangle(frame_img, (int(det.bb_left), int(det.bb_top)), (int(det.bb_left+det.bb_width), int(det.bb_top+det.bb_height)), (255, 0, 0), 2)
            if det.track_id > 0:
                if det.his_box is not None:
                    cv2.rectangle(frame_img, (int(det.his_box[0]), int(det.his_box[1])), (int(det.his_box[2]), int(det.his_box[3])), (0, 0, 255), 1)
                cv2.rectangle(frame_img, (int(det.bb_left)-5, int(det.bb_top)), (int(det.bb_left+det.bb_width)+5, int(det.bb_top+det.bb_height)), (0, 255, 0), 2)
                cv2.putText(frame_img, str(det.track_id), (int(det.bb_left), int(det.bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if det.his is not None:
                    start_point = det.his[0]
                    end_point = det.his[1]
                    cv2.arrowedLine(frame_img, (int(start_point[0]),int(start_point[1])), (int(end_point[0]),int(end_point[1])), (255,0,255), 2, tipLength=0.2)
                if det.his_pr is not None:
                    start_point = det.his_pr[0]
                    end_point = det.his_pr[1]
                    cv2.line(frame_img, (int(start_point[0]),int(start_point[1])), (int(end_point[0]),int(end_point[1])), (0,255,255), 2)                    
                # cv2.putText(frame_img,str(det.det_id), (int(det.bb_left+30), int(det.bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                # cv2.putText(frame_img,str(det.track_id_), (int(det.bb_left+50), int(det.bb_top)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                results.append(
                    f"{frame_id},{det.track_id},{det.bb_left:.2f},{det.bb_top:.2f},{det.bb_width:.2f},{det.bb_height:.2f},{det.conf:.2f},-1,-1,-1\n"
                )
        cv2.imshow("demo", frame_img)
        cv2.waitKey(1)
    with open(save_dir + f"{args.name}.txt", 'w') as f:
            f.writelines(results)
    cv2.destroyAllWindows()

parser = argparse.ArgumentParser(description='Process some arguments.')
parser.add_argument('--video', type=str, default = "demo/demo.mp4", help='video file name')
parser.add_argument('--cam_para', type=str, default = "camera_param/cam_para_test_", help='camera parameter file name')
parser.add_argument('--wx', type=float, default=5, help='wx')
parser.add_argument('--wy', type=float, default=5, help='wy')
parser.add_argument('--vmax', type=float, default=10, help='vmax')
parser.add_argument('--a', type=float, default=0.75, help='assignment threshold')
parser.add_argument('--cdt', type=float, default=3.0, help='coasted deletion time')
parser.add_argument('--high_score', type=float, default=0.5, help='high score threshold')
parser.add_argument('--conf_thresh', type=float, default=0.1, help='detection confidence threshold')
parser.add_argument('--name', type=str,default='0903_150000_151900', help='save results to project/name')
parser.add_argument('--path', type=str,default="/mnt/HDD8/tinery/MLSP/final_project/dataset/32_33_train_v2/train/images/", help='save results to project/name')
parser.add_argument('--model_config', type=str,default="fast_reid/configs/VeRi/sbs_R50-ibn.yml", help='save results to project/name')
parser.add_argument('--w_reid', type=str,default= "fast_reid/pretrained/model_0058_ft.pth", help='save results to project/name')
parser.add_argument('--w_detection', type=str,default='pretrained/yolov8x.pt', help='save results to project/name')

args = parser.parse_args()
 
main(args)



