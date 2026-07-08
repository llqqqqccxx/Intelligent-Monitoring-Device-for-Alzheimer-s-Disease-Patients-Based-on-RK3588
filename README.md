#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import subprocess
import sys
import socket
import urllib.request
import urllib.error
import re
import threading
import http.server
import socketserver
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from itertools import product as product
from math import ceil
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import time
from collections import deque
from datetime import datetime

# ====================== RKNN 推理引擎 ======================
from rknnlite.api import RKNNLite

# ====================== 基础配置 ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 屏幕分辨率
SCREEN_W = 1920
SCREEN_H = 1080

# 摄像头原始分辨率
ORIG_W = 640
ORIG_H = 480

# 全局显示尺寸（用于鼠标坐标映射）
g_display_w = 960
g_display_h = 720

# RKNN 模型文件路径
DET_RKNN = os.path.join(BASE_DIR, "RetinaFace_mobile320.rknn")
REC_RKNN = os.path.join(BASE_DIR, "w600k_mbf.rknn")

STANDBY_SCRIPT = os.path.join(BASE_DIR, "standby.py")
DOOR_CONFIG_FILE = os.path.join(BASE_DIR, "door_config.npy")

# 五视图特征文件
FEATURE_PATHS = {
    "front": os.path.join(BASE_DIR, "patient_feature_front.npy"),
    "left":  os.path.join(BASE_DIR, "patient_feature_left.npy"),
    "right": os.path.join(BASE_DIR, "patient_feature_right.npy"),
    "up":    os.path.join(BASE_DIR, "patient_feature_up.npy"),
    "down":  os.path.join(BASE_DIR, "patient_feature_down.npy"),
}

# ====================== 人脸识别核心阈值 ======================
FACE_DETECTION_INTERVAL = 2
FONT_SIZE = 18

ENTER_TRACK_WINDOW = 5
ENTER_TRACK_MIN_HIT = 2
FACE_POSE_MAX_DISTANCE = 0.8

# 追踪核心参数
POSE_PREDICT_ENABLE = True
NORMAL_LOST_MAX_FRAMES = 22
FALL_LOST_MAX_FRAMES = 450
POSE_MISS_TOLERANCE = 10
POSE_ENTER_VALID_THRESH = 2

POSITION_JUMP_THRESHOLD = 1.5
AREA_JUMP_THRESHOLD = 2.0
ASPECT_JUMP_THRESHOLD = 2.0
PERSON_SWITCH_CONFIRM_FRAMES = 60
MOTION_SMOOTH_FRAMES = 8

POSITION_RECOVER_THRESHOLD = 0.12
RECOVER_AREA_MAX_DIFF = 0.50
RECOVER_FACE_MANDATORY = True
FAST_FACE_SEARCH_FRAMES = 60

FACE_MIN_WIDTH = 40
FACE_MIN_HEIGHT = 40
MAX_YAW_DEGREE = 45
FACE_QUALITY_THRESHOLD = 0.50

STATE_IDLE = 0
STATE_MONITORING = 1
STATE_SAFE_MODE = 5
SAFE_MODE_ENTER_FRAMES = 15
SAFE_MODE_EXIT_FRAMES = 30

STRANGER_THRESHOLD = 0.25

BODY_BOX_PADDING_X = 15
BODY_BOX_PADDING_Y = 20
BOX_SMOOTH_ALPHA = 0.85

# 人脸验证参数
TRACK_DETECT_INTERVAL = 3
VERIFY_WINDOW_SIZE = 30
VERIFY_FAIL_THRESHOLD = 25
FACE_COMPLETENESS_SKIP_THRESH = 0.45

# 特征采集配置（每视角采集8帧，多帧平均）
NUM_CAPTURE_FRAMES = 8
CAPTURE_CONFIRM_FRAMES = 8

# ====================== 跌倒检测配置======================
FALL_AFTER_DISABLE_SECONDS = 0.0

STATE_FALL_CONFIRM = 2
STATE_FALL_ALERT = 3

# 强静态
FALL_STRONG_ASPECT = 1.05      
FALL_STRONG_HEAD_Y = 0.50     
FALL_STRONG_TORSO_ANGLE = 52   

# 弱静态
FALL_WEAK_HEAD_Y = 0.38    
FALL_WEAK_TORSO_ANGLE = 32     

# 动态下落过程阈值
FALL_PROCESS_HEAD_SPEED = 0.09     
FALL_PROCESS_ASPECT_CHANGE = 0.26  
FALL_PROCESS_VALID_WINDOW = 2.5   

# 辅助动态趋势阈值
FALL_HEAD_SPEED_THRESH = 0.06     
FALL_ASPECT_CHANGE_THRESH = 0.18  

# 跌倒后姿态恢复锁定
FALL_POST_ALERT_LOCK_SECONDS = 15
FALL_RECOVERY_ASPECT_THRESH = 0.70
FALL_RECOVERY_HEAD_Y_THRESH = 0.50
FALL_RECOVERY_CONFIRM_FRAMES = 10

FALL_CONFIRM_FRAMES = 5
FINAL_CONFIRM_SECONDS = 2.0
FALL_RATIO_THRESHOLD = 0.50
FALL_COOLDOWN_SECONDS = 5
FALL_ALERT_DURATION_SECONDS = 2.0

HISTORY_FRAMES = 10

POSE_LOST_CONFIRM_FRAMES = 8
POSE_LOST_MIN_FALL_FRAMES = 2
POSE_LOST_FORCE_CONFIRM_FRAMES = 15
POSE_LOST_LOCK_FRAMES = 12
FALL_CONFIRM_IRREVERSIBLE_THRESHOLD = 5
FALL_CONFIRM_RECOVERY_THRESHOLD = 6

# 跌倒后禁用久坐时长
SEDENTARY_DISABLE_AFTER_FALL_SECONDS = 30

# ====================== 门线走失配置 ======================
DEFAULT_DOOR_POINTS = np.array([
    [200, 0],
    [440, 0],
    [440, 480],
    [200, 480]
], dtype=np.int32)

DOOR_PEAK_RATIO_THRESH = 0.65
DOOR_LAST_FRAMES_AVG_THRESH = 0.10
DOOR_RATIO_WINDOW = 30
WANDER_ALERT_DISPLAY_FRAMES = 180
DOOR_LINE_COLOR = (0, 165, 255)
WANDER_COOLDOWN_SECONDS = 10

POSE_KEYPOINT_VIS_THRESH = 0.40
POSE_MIN_VALID_KEYPOINTS = 4
POSE_VALID_CONFIRM_FRAMES = 1
WANDER_JUDGE_DELAY_FRAMES = 10

# ====================== 久坐检测配置======================
SEDENTARY_COOLDOWN_SECONDS = 30
SITTING_TORSO_RATIO_THRESH = 0.65
SITTING_UPPER_BODY_RATIO_THRESH = 0.78
SITTING_HEAD_Y_THRESH = 0.65
SITTING_CONFIRM_FRAMES = 3
SEDENTARY_THRESHOLD_SECONDS = 30
SEDENTARY_MOVE_THRESHOLD = 0.15
SEDENTARY_DEBUG = False
SEDENTARY_LOST_MAX_FRAMES = 120

# ====================== ntfy 通知配置 ======================
NTFY_TOPIC = "jiashu_arz"                     # ntfy 主题
NTFY_SERVER = "https://ntfy.sh"               # ntfy 服务器

# ====================== 本地 HTTP / MJPG 端口 ======================
HTTP_SERVER_PORT = 8080          # 提供视频文件下载
MJPEG_PORT = 8081                # 提供实时流

# ====================== 久坐通知冷却（独立） ======================
SEDENTARY_NTFY_COOLDOWN = 180    # 两次久坐通知间隔（秒）

# ====================== 视频录制配置 ======================
VIDEO_FPS = 15
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_SAVE_FOLDER = "alert_videos"
FALL_PRE_RECORD_SECONDS = 4.0
FALL_POST_RECORD_SECONDS = 4.0
WANDER_PRE_RECORD_SECONDS = 6.0
WANDER_POST_RECORD_SECONDS = 2.0

_calibrating_door = False
_calibration_points = []

# ====================== MJPG 流全局变量 ======================
latest_jpg = None
jpg_lock = threading.Lock()

# 主程序退出时用于关闭 cloudflared 进程
cloudflared_proc = None

# ====================== 工具函数 ======================
def put_chinese_text(img, text, pos, color=(0, 255, 0), font_size=FONT_SIZE):
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", font_size)
    except:
        font = ImageFont.load_default()
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(pos, text, fill=color, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def normalize_feature(feature):
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 1e-6 else feature

def get_dynamic_enter_threshold(face_width):
    if face_width > 120:
        return 0.43
    elif 100 <= face_width <= 120:
        return 0.40
    elif 80 <= face_width < 100:
        return 0.37
    elif 70 <= face_width < 80:
        return 0.35
    elif 60 <= face_width < 70:
        return 0.33
    elif 40 <= face_width < 60:
        return 0.27
    else:
        return 1.0

def get_dynamic_verify_threshold(face_width):
    if face_width > 120:
        return 0.39
    elif 100 <= face_width <= 120:
        return 0.36
    elif 80 <= face_width < 100:
        return 0.32
    elif 70 <= face_width < 80:
        return 0.30
    elif 60 <= face_width < 70:
        return 0.27
    elif 40 <= face_width < 60:
        return 0.22
    else:
        return 1.0

def align_face(img_bgr, landmarks_5, target_size=(112, 112)):
    src_pts = np.array(landmarks_5, dtype=np.float32)
    dst_pts = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None:
        return cv2.resize(img_bgr, target_size)
    aligned_face = cv2.warpAffine(img_bgr, M, target_size, borderValue=(0, 0, 0))
    return aligned_face

def is_face_complete_by_bbox(bbox, frame_width, frame_height, loose_mode=False):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w < FACE_MIN_WIDTH or h < FACE_MIN_HEIGHT: return False
    aspect = w / h
    if loose_mode:
        if aspect < 0.35 or aspect > 2.0: return False
        if (bbox[0] < 2 or bbox[1] < 2 or
            bbox[2] > frame_width - 2 or bbox[3] > frame_height - 2): return False
    else:
        if aspect < 0.5 or aspect > 1.6: return False
        if (bbox[0] < 5 or bbox[1] < 5 or
            bbox[2] > frame_width - 5 or bbox[3] > frame_height - 5): return False
    return True

def calc_face_completeness(bbox, frame_width, frame_height):
    x1, y1, x2, y2 = bbox
    face_w = x2 - x1
    face_h = y2 - y1
    if face_w <= 0 or face_h <= 0:
        return 0.0
    left_cut = max(0, 0 - x1) / face_w
    right_cut = max(0, x2 - frame_width) / face_w
    top_cut = max(0, 0 - y1) / face_h
    bottom_cut = max(0, y2 - frame_height) / face_h
    boundary_score = 1.0 - (left_cut + right_cut + top_cut + bottom_cut) / 4
    aspect = face_w / face_h
    aspect_score = max(0.0, 1.0 - abs(aspect - 0.7) / 0.7)
    return max(0.0, min(1.0, boundary_score * 0.8 + aspect_score * 0.2))

def is_face_frontal(landmarks_5):
    if landmarks_5 is None or len(landmarks_5) != 5: return 0.0
    pts = np.array(landmarks_5)
    le, re, nose, lm, rm = pts[0], pts[1], pts[2], pts[3], pts[4]
    eye_dy = abs(le[1] - re[1])
    eye_dx = abs(le[0] - re[0]) + 1e-6
    eye_score = max(0.0, 1.0 - (eye_dy / eye_dx) * 5)
    eye_mid_x = (le[0] + re[0]) / 2
    nose_score = max(0.0, 1.0 - abs(nose[0] - eye_mid_x) / eye_dx * 3)
    mouth_mid_x = (lm[0] + rm[0]) / 2
    mouth_score = max(0.0, 1.0 - abs(mouth_mid_x - nose[0]) / eye_dx * 3)
    return (eye_score + nose_score + mouth_score) / 3.0

def estimate_yaw(landmarks_5):
    if landmarks_5 is None or len(landmarks_5) != 5: return 90
    pts = np.array(landmarks_5)
    le, re, nose = pts[0], pts[1], pts[2]
    eye_mid = (le + re) / 2
    offset_ratio = (nose[0] - eye_mid[0]) / (abs(re[0] - le[0]) + 1e-6)
    yaw = offset_ratio * 90
    return abs(yaw)

def get_face_view(landmarks_5):
    if landmarks_5 is None or len(landmarks_5) != 5:
        return "front"
    pts = np.array(landmarks_5)
    le, re, nose = pts[0], pts[1], pts[2]
    eye_mid_x = (le[0] + re[0]) / 2
    eye_width = abs(re[0] - le[0]) + 1e-6
    offset = (nose[0] - eye_mid_x) / eye_width
    if abs(offset) < 0.35:
        return "front"
    elif offset < -0.35:
        return "left"
    else:
        return "right"

def calc_view_matched_sim(emb, landmarks_5, feature_dict):
    max_sim = 0.0
    for feat in feature_dict.values():
        s = float(np.dot(emb, feat))
        if s > max_sim:
            max_sim = s
    return max_sim, "best"

# ====================== 全屏自适应拉伸函数 ======================
def fullscreen_resize(frame, window_name="Face + Pose Tracking"):
    global g_display_w, g_display_h
    try:
        fullscreen_prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
        if fullscreen_prop == cv2.WINDOW_FULLSCREEN:
            g_display_w, g_display_h = SCREEN_W, SCREEN_H
            return cv2.resize(frame, (SCREEN_W, SCREEN_H))
        else:
            win_w = int(cv2.getWindowProperty(window_name, cv2.WND_PROP_WIDTH))
            win_h = int(cv2.getWindowProperty(window_name, cv2.WND_PROP_HEIGHT))
            if win_w > 0 and win_h > 0:
                g_display_w, g_display_h = win_w, win_h
                return cv2.resize(frame, (win_w, win_h))
    except:
        pass
    return frame

# ====================== 门线工具函数（全屏标定坐标映射）======================
def calc_body_door_ratio(body_bbox, door_polygon):
    x1, x2, y1, y2 = body_bbox
    bw = int(x2 - x1)
    bh = int(y2 - y1)
    if bw <= 0 or bh <= 0:
        return 0.0
    shifted_door = door_polygon - np.array([x1, y1], dtype=np.int32)
    mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(mask, [shifted_door], 255)
    inside_pixels = cv2.countNonZero(mask)
    return inside_pixels / (bw * bh)

def is_point_in_door(point, door_polygon):
    return cv2.pointPolygonTest(door_polygon, point, False) >= 0

def _door_mouse_callback(event, x, y, flags, param):
    global _calibrating_door, _calibration_points, g_display_w, g_display_h
    if not _calibrating_door:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        # 将显示坐标映射回原始帧坐标
        scale_x = ORIG_W / g_display_w if g_display_w > 0 else 1.0
        scale_y = ORIG_H / g_display_h if g_display_h > 0 else 1.0
        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)
        orig_x = max(0, min(ORIG_W - 1, orig_x))
        orig_y = max(0, min(ORIG_H - 1, orig_y))
        if len(_calibration_points) < 4:
            _calibration_points.append([orig_x, orig_y])
            print(f"[标定] 已选第{len(_calibration_points)}/4个点（原始坐标: ({orig_x}, {orig_y})）")
            if len(_calibration_points) == 4:
                _calibrating_door = False
                print("[标定] ✅ 4个点采集完成，门线已保存")

def load_door_config():
    if os.path.exists(DOOR_CONFIG_FILE):
        try:
            points = np.load(DOOR_CONFIG_FILE)
            if points.shape == (4, 2):
                print("✅ 已加载本地门线配置")
                return points.astype(np.int32)
        except:
            pass
    print("ℹ️  使用默认门线配置，按'm'键可标定")
    return DEFAULT_DOOR_POINTS.copy()

def save_door_config(points):
    np.save(DOOR_CONFIG_FILE, points)
    print(f"💾 门线配置已保存到: {DOOR_CONFIG_FILE}")

# ====================== 视频录制工具 ======================
def create_alert_video_folder():
    if not os.path.exists(VIDEO_SAVE_FOLDER):
        os.makedirs(VIDEO_SAVE_FOLDER)
    for sub in ["fall", "wander"]:
        p = os.path.join(VIDEO_SAVE_FOLDER, sub)
        if not os.path.exists(p):
            os.makedirs(p)

def get_alert_video_path(event_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = "fall" if event_type == "fall" else "wander"
    return os.path.join(VIDEO_SAVE_FOLDER, sub_dir, f"{event_type}_{timestamp}.avi")

# ====================== RetinaFace 后处理 ======================
def PriorBox(image_size):
    anchors = []
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]
    for k, f in enumerate(feature_maps):
        min_sizes_k = min_sizes[k]
        for i in range(f[0]):
            for j in range(f[1]):
                for min_size in min_sizes_k:
                    s_kx = min_size / image_size[1]
                    s_ky = min_size / image_size[0]
                    dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
    return np.array(anchors).reshape(-1, 4)

def box_decode(loc, priors):
    variances = [0.1, 0.2]
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    ), axis=1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def nms(dets, thresh):
    x1 = dets[:, 0]; y1 = dets[:, 1]; x2 = dets[:, 2]; y2 = dets[:, 3]; scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def dedup_faces(faces_with_sim, iou_thresh=0.3):
    if not faces_with_sim: return []
    boxes = np.array([f[0] for f in faces_with_sim])
    sims = np.array([f[1] for f in faces_with_sim])
    dets = np.hstack((boxes, sims.reshape(-1, 1)))
    keep_idx = nms(dets, iou_thresh)
    return [faces_with_sim[i] for i in keep_idx]

# ====================== RKNN 人脸引擎 ======================
class RKNNFace:
    USE_BUILTIN_PREPROCESS_DET = True
    USE_BUILTIN_PREPROCESS_REC = True

    def __init__(self, det_rknn_path, rec_rknn_path):
        self.det_rknn = RKNNLite()
        ret = self.det_rknn.load_rknn(det_rknn_path)
        if ret != 0:
            raise RuntimeError(f"加载检测 RKNN 模型失败: {det_rknn_path}")
        ret = self.det_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError("初始化检测 RKNN runtime 失败")

        self.rec_rknn = RKNNLite()
        ret = self.rec_rknn.load_rknn(rec_rknn_path)
        if ret != 0:
            raise RuntimeError(f"加载识别 RKNN 模型失败: {rec_rknn_path}")
        ret = self.rec_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        if ret != 0:
            raise RuntimeError("初始化识别 RKNN runtime 失败")

        self.det_input_size = (320, 320)
        self.rec_input_size = (112, 112)
        self.priors = PriorBox(self.det_input_size)
        self.nms_threshold = 0.45
        self.confidence_threshold = 0.5

    def _det_preprocess(self, img_bgr):
        h, w = img_bgr.shape[:2]
        target_w, target_h = self.det_input_size
        r = min(target_w / w, target_h / h)
        new_w = int(w * r); new_h = int(h * r)
        resized = cv2.resize(img_bgr, (new_w, new_h))
        pad_w = target_w - new_w; pad_h = target_h - new_h
        pad_left = pad_w // 2; pad_top = pad_h // 2
        padded = np.ones((target_h, target_w, 3), dtype=np.uint8) * 114
        padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
        blob = np.expand_dims(padded, axis=0).astype(np.uint8)  # (1, H, W, 3)
        return blob, r, pad_left, pad_top

    def detect(self, img_bgr):
        img_h, img_w = img_bgr.shape[:2]
        blob, scale, pad_left, pad_top = self._det_preprocess(img_bgr)
        outputs = self.det_rknn.inference(inputs=[blob])
        if outputs is None or len(outputs) != 3:
            return []

        loc = outputs[0].squeeze(0)
        conf = outputs[1].squeeze(0)
        landmarks_raw = outputs[2].squeeze(0)
        landmarks_raw = landmarks_raw.reshape(-1, 5, 2)  # 确保形状正确

        scores = conf[:, 1]
        boxes = box_decode(loc, self.priors)
        boxes[:, 0] = boxes[:, 0] * self.det_input_size[0] - pad_left
        boxes[:, 1] = boxes[:, 1] * self.det_input_size[1] - pad_top
        boxes[:, 2] = boxes[:, 2] * self.det_input_size[0] - pad_left
        boxes[:, 3] = boxes[:, 3] * self.det_input_size[1] - pad_top
        boxes /= scale
        boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, img_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, img_h)

        inds = np.where(scores > self.confidence_threshold)[0]
        if len(inds) == 0: return []
        boxes = boxes[inds]; scores = scores[inds]; landmarks = landmarks_raw[inds]
        priors = self.priors[inds]
        variances = [0.1, 0.2]

        landmarks[:, :, 0] = priors[:, None, 0] + landmarks[:, :, 0] * variances[0] * priors[:, None, 2]
        landmarks[:, :, 1] = priors[:, None, 1] + landmarks[:, :, 1] * variances[0] * priors[:, None, 3]
        landmarks[:, :, 0] = landmarks[:, :, 0] * self.det_input_size[0] - pad_left
        landmarks[:, :, 1] = landmarks[:, :, 1] * self.det_input_size[1] - pad_top
        landmarks /= scale
        landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, img_w)
        landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, img_h)

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, self.nms_threshold)
        if len(keep) == 0: return []
        final_boxes = boxes[keep]
        final_landmarks = landmarks[keep]
        return [(final_boxes[i].tolist(), final_landmarks[i].tolist()) for i in range(len(keep))]

    def extract(self, img_bgr, bbox, landmarks_5=None):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = img_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1: return None

        if landmarks_5 is not None and len(landmarks_5) == 5:
            aligned = align_face(img_bgr, landmarks_5, self.rec_input_size)
            face_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        else:
            face = img_bgr[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_rgb = cv2.resize(face_rgb, self.rec_input_size)

        blob = face_rgb.astype(np.uint8)
        blob = np.expand_dims(blob, axis=0)  # (1, 112, 112, 3)
        outputs = self.rec_rknn.inference(inputs=[blob])
        if outputs is None or len(outputs) == 0 or outputs[0] is None:
            return None
        emb = outputs[0].flatten()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def release(self):
        if hasattr(self, 'det_rknn') and self.det_rknn is not None:
            self.det_rknn.release()
        if hasattr(self, 'rec_rknn') and self.rec_rknn is not None:
            self.rec_rknn.release()

# ====================== 卡尔曼滤波 ======================
class Kalman2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.eye(4, dtype=np.float32)
        self.kf.transitionMatrix[0, 2] = 1.0
        self.kf.transitionMatrix[1, 3] = 1.0
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.15
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def update(self, x, y):
        if not self.initialized:
            self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True
            return x, y
        measured = np.array([[x], [y]], dtype=np.float32)
        self.kf.correct(measured)
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def predict(self):
        if not self.initialized:
            return 0.5, 0.5
        pred = self.kf.predict()
        return float(pred[0, 0]), float(pred[1, 0])

    def reset(self):
        self.initialized = False

# ====================== 特征采集模块（5视角，每视角8帧）======================
def feature_capture_mode(face_engine, cap):
    view_order = ["front", "left", "right", "up", "down"]
    view_labels = {
        "front": "正脸",
        "left":  "左侧脸",
        "right": "右侧脸",
        "up":    "抬头",
        "down":  "低头",
    }
    view_yaw_range = {
        "front": (0, 35),
        "left":  (25, 60),
        "right": (25, 60),
        "up":    (0, 35),
        "down":  (0, 35),
    }
    WINDOW_NAME = "Face + Pose Tracking"

    print("\n" + "="*60)
    print("  进入人脸特征采集模式（共5个视角）")
    print("  采集顺序：正脸 → 左侧脸 → 右侧脸 → 抬头 → 低头")
    print("  每个视角按 s 键开始采集，按 ESC 取消")
    print("="*60)

    for view_idx, view_name in enumerate(view_order):
        view_label = view_labels[view_name]
        save_path = FEATURE_PATHS[view_name]
        min_yaw, max_yaw = view_yaw_range[view_name]

        print(f"\n【第 {view_idx+1}/5 个视角】准备采集：{view_label}")
        print("请面对镜头保持对应姿势，按 s 键开始采集...")

        while True:
            ret, frame = cap.read()
            if not ret: return
            frame = put_chinese_text(frame, f"第 {view_idx+1}/5 步：采集{view_label}", (15, 40), (0, 255, 255), 22)
            frame = put_chinese_text(frame, "按 s 键开始采集，按 ESC 取消", (15, 80), (255, 255, 255), 16)
            # 全屏模式使用自动拉伸显示
            cv2.imshow(WINDOW_NAME, fullscreen_resize(frame, WINDOW_NAME))
            key = cv2.waitKey(1) & 0xFF
            # ---------- 全屏切换 ----------
            if key == ord('f'):
                prop = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
                if prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # ----------------------------
            if key == 27:
                print("已取消采集")
                return
            if key == ord('s'):
                break

        print(f"正在采集 {view_label}，请保持稳定...")
        stable_count = 0
        last_bbox_area = 0
        feature_buffer = []
        total_needed = NUM_CAPTURE_FRAMES

        while len(feature_buffer) < total_needed:
            ret, frame = cap.read()
            if not ret: break
            dets = face_engine.detect(frame)
            h, w = frame.shape[:2]

            if dets:
                best_det = max(dets, key=lambda d: (d[0][2]-d[0][0]) * (d[0][3]-d[0][1]))
                bbox, landmarks = best_det
                x1, y1, x2, y2 = [int(v) for v in bbox]
                yaw = estimate_yaw(landmarks)

                pose_ok = False
                if view_name in ("front", "up", "down"):
                    pose_ok = yaw <= max_yaw
                else:
                    pose_ok = yaw >= min_yaw and yaw <= max_yaw

                completeness = calc_face_completeness(bbox, w, h)
                quality_ok = completeness >= FACE_QUALITY_THRESHOLD

                if pose_ok and quality_ok and is_face_complete_by_bbox(bbox, w, h, loose_mode=False):
                    area = (x2-x1) * (y2-y1)
                    if abs(area - last_bbox_area) / (area+1e-6) < 0.25:
                        stable_count += 1
                    else:
                        stable_count = 0
                    last_bbox_area = area

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    frame = put_chinese_text(frame, f"采集进度：{len(feature_buffer)}/{total_needed}", (x1, y1-35), (0, 255, 0), 16)
                    frame = put_chinese_text(frame, f"角度：{yaw:.1f}° | 质量：{completeness:.2f}", (x1, y1-60), (255, 255, 0), 14)

                    if stable_count >= CAPTURE_CONFIRM_FRAMES and stable_count % 2 == 0:
                        emb = face_engine.extract(frame, bbox, landmarks)
                        if emb is not None:
                            feature_buffer.append(emb)
                else:
                    stable_count = 0
                    if not pose_ok:
                        frame = put_chinese_text(frame, "请调整脸部角度", (x1, y1-30), (0, 0, 255), 16)
            else:
                stable_count = 0
                frame = put_chinese_text(frame, "未检测到人脸，请正对镜头", (15, 120), (0, 0, 255), 18)

            frame = put_chinese_text(frame, f"当前采集：{view_label} ({view_idx+1}/5)", (15, 80), (0, 255, 255), 20)
            cv2.imshow(WINDOW_NAME, fullscreen_resize(frame, WINDOW_NAME))
            key = cv2.waitKey(1) & 0xFF
            # ---------- 全屏切换 ----------
            if key == ord('f'):
                prop = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
                if prop == cv2.WINDOW_FULLSCREEN:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                else:
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # ----------------------------
            if key == 27:
                print("用户取消采集")
                return

        if len(feature_buffer) >= total_needed:
            avg_feat = np.mean(feature_buffer, axis=0)
            avg_feat = avg_feat / np.linalg.norm(avg_feat)
            np.save(save_path, avg_feat)
            print(f"✅ {view_label} 采集完成")
            print(f"   采集帧数: {len(feature_buffer)}, 特征模长: {np.linalg.norm(avg_feat):.4f}")

            for _ in range(20):
                ret, frame = cap.read()
                if not ret: break
                frame = put_chinese_text(frame, f"{view_label} 采集完成！", (15, 150), (0, 255, 0), 24)
                cv2.imshow(WINDOW_NAME, fullscreen_resize(frame, WINDOW_NAME))
                key = cv2.waitKey(30) & 0xFF
                if key == ord('f'):
                    prop = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
                    if prop == cv2.WINDOW_FULLSCREEN:
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    else:
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                if key == 27:
                    break
        else:
            print(f"❌ {view_label} 采集失败，人脸丢失或姿态不符")
            return

    print("\n🎉 全部5视角特征采集完成！")

# ====================== 网络功能函数 ======================
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.254.254.254', 1))
        ip = s.getsockname()[0]
    except:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def start_local_http_server(port=HTTP_SERVER_PORT):
    video_dir = os.path.join(BASE_DIR, VIDEO_SAVE_FOLDER)
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=video_dir, **kwargs)
    httpd = socketserver.TCPServer(("", port), Handler)
    local_ip = get_local_ip()
    print(f"📡 视频文件服务: http://{local_ip}:{port}")
    httpd.serve_forever()

def send_ntfy_alert(event_type, video_path=None):
    """
    发送 ntfy 通知。
    event_type: "fall", "wander", "sedentary", "manual"
    video_path: 如果提供，会在消息中包含局域网下载链接
    """
    local_ip = get_local_ip()
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if event_type in ("fall", "wander", "manual"):
        if video_path and os.path.exists(video_path):
            video_relpath = os.path.relpath(video_path, os.path.join(BASE_DIR, VIDEO_SAVE_FOLDER))
            video_relpath = video_relpath.replace('\\', '/')
            video_url = f"http://{local_ip}:{HTTP_SERVER_PORT}/{video_relpath}"
            message = f"🚨 检测到患者{event_type}！\n时间: {timestamp_str}\n视频: {video_url}"
            click_url = video_url
        else:
            message = f"🚨 检测到患者{event_type}！\n时间: {timestamp_str}\n（未生成视频）"
            click_url = ""
    elif event_type == "sedentary":
        message = f"🪑 久坐提醒\n患者已连续久坐超过30秒，请及时起身活动。\n时间: {timestamp_str}"
        click_url = ""

    # 尝试 curl 发送，失败则用 urllib
    success = False
    try:
        cmd = ["curl", "-H", "Tags: warning,fall", "-d", message]
        if click_url:
            cmd.insert(2, "-H"); cmd.insert(3, f"Click: {click_url}")
        cmd.append(f"{NTFY_SERVER}/{NTFY_TOPIC}")
        subprocess.run(cmd, capture_output=True, timeout=10, check=True)
        success = True
    except:
        pass

    if not success:
        try:
            ntfy_req = urllib.request.Request(
                f"{NTFY_SERVER}/{NTFY_TOPIC}",
                data=message.encode("utf-8"),
                headers={"Tags": "warning,fall", "Click": click_url}
            )
            urllib.request.urlopen(ntfy_req, timeout=10)
            success = True
        except Exception as e:
            print(f"❌ ntfy 通知发送失败: {e}")

    if success:
        print(f"✅ 已发送 ntfy 通知: {event_type}")
    else:
        print("❌ ntfy 通知发送失败")

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class MjpegHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/stream' or self.path == '/mjpeg':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Connection', 'close')
            self.end_headers()
            print(f"📹 客户端 {self.client_address} 连接 MJPG 流")
            try:
                while True:
                    with jpg_lock:
                        data = latest_jpg
                    if data is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(data)}\r\n'.encode())
                    self.wfile.write(b'\r\n')
                    self.wfile.write(data)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                print(f"📴 客户端 {self.client_address} 断开")
            except Exception as e:
                print(f"⚠️ MJPG 错误: {e}")
        elif self.path == '/':
            local_ip = get_local_ip()
            html = f"""<html><body>
            <h2>患者监护实时画面</h2>
            <img src="/stream" width="640" height="480" />
            <p>视频文件目录: <a href="http://{local_ip}:{HTTP_SERVER_PORT}">点此访问</a></p>
            </body></html>"""
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

def start_mjpeg_server(port=MJPEG_PORT):
    try:
        server = ThreadingHTTPServer(('0.0.0.0', port), MjpegHandler)
        local_ip = get_local_ip()
        print(f"📡 MJPG 实时流: http://{local_ip}:{port}/stream")
        server.serve_forever()
    except Exception as e:
        print(f"❌ MJPG 服务器启动失败: {e}")

def start_cloudflared_tunnel(local_port=MJPEG_PORT):
    """后台运行 cloudflared，自动提取并打印公网地址"""
    cmd = ["cloudflared", "tunnel", "--protocol", "http2",
           "--url", f"http://localhost:{local_port}"]
    global cloudflared_proc
    try:
        cloudflared_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
    except FileNotFoundError:
        print("⚠️ 未找到 cloudflared，跳过公网隧道")
        return

    def reader():
        found_url = False
        for line in cloudflared_proc.stdout:
            print(f"[cloudflared] {line.rstrip()}")
            match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
            if match and not found_url:
                found_url = True
                url = match.group(0)
                print(f"\n🌐 公网实时画面: {url}/stream")
                print(f"📁 视频下载目录: {url.replace(f':{local_port}', f':{HTTP_SERVER_PORT}')}/\n")

    threading.Thread(target=reader, daemon=True).start()
# ====================== 主程序 ======================
def main():
    global _calibrating_door, _calibration_points, DOOR_POINTS, latest_jpg, last_recorded_video_path

    DOOR_POINTS = load_door_config()

    feature_dict = {}
    print("加载患者特征库...")
    for name, path in FEATURE_PATHS.items():
        if os.path.exists(path):
            feat = np.load(path)
            feat = feat / np.linalg.norm(feat)
            feature_dict[name] = feat
            print(f"  ✅ 已加载: {name}")
        else:
            print(f"  ⚠️  未找到: {name}")
    if not feature_dict:
        print("❌ 未加载到任何特征文件，请先按's'键采集患者特征")

    print("\n正在加载RKNN人脸模型...")
    try:
        face_engine = RKNNFace(DET_RKNN, REC_RKNN)
        print("✅ RKNN人脸模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )
    mp_drawing = mp.solutions.drawing_utils

    # 摄像头配置（RK3588 使用 /dev/video21）
    cap = cv2.VideoCapture(21, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    if not cap.isOpened():
        print("❌ 无法打开摄像头 /dev/video21")
        return

    create_alert_video_folder()

    # ---------- 启动本地 HTTP 文件服务器（用于视频下载）----------
    threading.Thread(target=start_local_http_server, daemon=True).start()

    # ---------- 启动 MJPG 流服务器 ----------
    threading.Thread(target=start_mjpeg_server, daemon=True).start()

    # ---------- 启动 Cloudflare 公网隧道 ----------
    start_cloudflared_tunnel()

    WINDOW_NAME = "Face + Pose Tracking"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 720)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)  # 解除宽高比锁定
    cv2.setMouseCallback(WINDOW_NAME, _door_mouse_callback)
    frame_counter = 0

    cached_valid_face_count = 0
    cached_total_face_count = 0
    cached_has_patient = False
    cached_patient_face = None
    cached_patient_similarity = 0.0
    cached_patient_face_center_norm = None
    cached_patient_face_width = 0
    fast_enter_streak = 0

    cached_nearest_face_exist = False
    cached_nearest_face_sim = 0.0
    cached_nearest_face_width = 0
    cached_nearest_face_bbox = None

    sim_smooth_buffer = deque(maxlen=12)
    enter_track_buffer = deque(maxlen=ENTER_TRACK_WINDOW)

    last_patient_position = None
    last_patient_body_area = None
    last_patient_body_aspect = None
    motion_history = deque(maxlen=MOTION_SMOOTH_FRAMES)

    person_switch_abnormal_count = 0

    pose_tracking_enabled = False
    pose_was_lost = False
    pose_predict_mode = False
    last_known_patient_position = None
    last_known_body_area = None

    has_patient_face = False
    patient_bbox = None
    patient_sim = 0.0
    face_detected = False
    last_body_bbox = None
    pose_miss_count = 0
    pose_valid_streak = 0

    fast_face_search = False
    fast_search_frames = 0
    fast_search_printed = False
    head_y_history = []

    current_state = STATE_IDLE
    backup_tracking_enabled = False
    backup_patient_position = None
    backup_patient_body_area = None
    backup_patient_body_aspect = None
    backup_fall_state = STATE_IDLE

    safe_mode_enter_counter = 0
    safe_mode_exit_counter = 0

    idle_printed = False
    tracking_lost_printed = False
    pose_lost_printed = False
    safe_mode_printed = False
    monitor_printed = False

    kf_pose = Kalman2D()
    pose_lost_frames = 0
    stop_verify_buffer = deque(maxlen=VERIFY_WINDOW_SIZE)

    # 跌倒检测状态
    fall_state = STATE_IDLE
    fall_count = 0
    fall_frames_in_confirm = 0
    total_frames_in_confirm = 0
    fall_recovery_count = 0
    consecutive_normal_frames = 0
    fall_locked = False
    pose_lost_fall_count = 0
    has_valid_fall_before_lost = False
    last_fall_detect_time = 0
    last_fall_alert_start_time = 0
    last_fall_disable_end_time = 0.0
    final_confirm_start_time = 0
    aspect_ratio_history = []
    fall_alert_triggered = False

    fall_process_detected = False
    fall_process_last_time = 0.0
    post_fall_lock = False
    post_fall_lock_until = 0.0
    recovery_streak = 0

    sedentary_disable_until_time = 0.0

    # 患者在家状态
    patient_at_home = False
    home_state_printed = False

    door_ratio_history = deque(maxlen=DOOR_RATIO_WINDOW)
    last_body_door_ratio = 0.0
    last_body_in_door = False
    wander_judged = False
    wander_alert_active = False
    wander_alert_remain = 0
    wander_printed = False
    last_patient_center_pixel = None
    wander_alert_triggered = False
    last_wander_alert_time = 0
    wander_predict_frames = 0

    sedentary_start_time = None
    accumulated_sedentary_time = 0.0
    last_sedentary_alert_time = 0
    last_sedentary_position = None
    sedentary_alert_display = False
    sedentary_display_remain = 0
    sedentary_lost_frames = 0
    last_debug_print_time = 0
    last_ref_type = None
    last_sitting_state = None
    torso_ratio = None
    current_sedentary_seconds = 0.0
    sitting_streak = 0
    last_raw_sitting = True

    # 视频录制状态
    frame_buffer = []
    is_recording = False
    video_writer = None
    recording_start_time = 0
    recording_post_duration = 0
    recording_event_type = ""
    last_recorded_video_path = None

    print("\n🎥 监护系统启动成功 (RK3588)")
    print("初始状态：患者不在家")
    print("操作说明：按's'采集特征 | 按'm'标定门线 | 按'q'返回待机 | 按'ESC'退出 | 按'f'全屏")
    print("-" * 60)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # ---- 更新 MJPG 流（每帧都做）----
        small_frame = cv2.resize(frame, (320, 240))
        _, jpg = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        with jpg_lock:
            latest_jpg = jpg.tobytes()

        h, w = frame.shape[:2]
        frame_counter += 1
        current_time = time.time()

        # 预录环形缓冲区
        frame_buffer.append((current_time, frame.copy()))
        max_pre_record = max(FALL_PRE_RECORD_SECONDS, WANDER_PRE_RECORD_SECONDS)
        while len(frame_buffer) > 0 and current_time - frame_buffer[0][0] > max_pre_record:
            frame_buffer.pop(0)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[INFO] 返回待机界面...")
            cap.release()
            cv2.destroyAllWindows()
            pose.close()
            face_engine.release()
            if video_writer is not None:
                video_writer.release()
            if os.path.exists(STANDBY_SCRIPT):
                subprocess.Popen([sys.executable, STANDBY_SCRIPT], cwd=BASE_DIR)
            else:
                print(f"[警告] 未找到待机程序: {STANDBY_SCRIPT}")
            return

        if key == ord('s'):
            feature_capture_mode(face_engine, cap)
            feature_dict.clear()
            sim_smooth_buffer.clear()
            for name, path in FEATURE_PATHS.items():
                if os.path.exists(path):
                    feat = np.load(path)
                    feat = feat / np.linalg.norm(feat)
                    feature_dict[name] = feat
                    print(f"  ✅ 已重新加载: {name}")
            # 重置状态
            pose_tracking_enabled = False
            pose_was_lost = False
            pose_predict_mode = False
            has_patient_face = False
            last_patient_position = None
            last_patient_body_area = None
            last_patient_body_aspect = None
            last_known_patient_position = None
            last_known_body_area = None
            current_state = STATE_IDLE
            fall_state = STATE_IDLE
            fall_process_detected = False
            fall_process_last_time = 0.0
            post_fall_lock = False
            recovery_streak = 0
            stop_verify_buffer.clear()
            door_ratio_history.clear()
            wander_judged = False
            wander_printed = False
            wander_alert_active = False
            wander_alert_triggered = False
            idle_printed = False
            last_body_bbox = None
            pose_valid_streak = 0
            fast_search_printed = False
            enter_track_buffer.clear()
            fast_enter_streak = 0
            person_switch_abnormal_count = 0
            motion_history.clear()
            sedentary_start_time = None
            accumulated_sedentary_time = 0.0
            last_sedentary_position = None
            last_ref_type = None
            last_sitting_state = None
            sedentary_alert_display = False
            sedentary_lost_frames = 0
            current_sedentary_seconds = 0.0
            sitting_streak = 0
            last_raw_sitting = True
            wander_predict_frames = 0
            pose_lost_frames = 0
            last_fall_disable_end_time = 0.0
            sedentary_disable_until_time = 0.0
            # 不影响在家状态
            continue

        if key == ord('m'):
            _calibrating_door = True
            _calibration_points = []
            print("\n[标定] 进入门线标定模式，请依次点击4个点围成门区域")
            print("[标定] 顺序：左上 → 右上 → 右下 → 左下")
            continue
        if key == ord('f'):
            prop = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
            if prop == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            else:
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if key == 27:
            break

        if len(_calibration_points) == 4 and not _calibrating_door:
            DOOR_POINTS = np.array(_calibration_points, dtype=np.int32)
            save_door_config(DOOR_POINTS)
            _calibration_points = []
            door_ratio_history.clear()
            wander_judged = False
            wander_printed = False

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        has_pose_raw = (results.pose_landmarks is not None)

        current_pose_center = None
        all_pose_points = []
        pose_valid_this_frame = False
        torso_ratio = None
        current_body_area = None
        current_body_aspect = None
        body_h_norm = 0.0
        lower_body_visible_count = 0
        torso_angle = 0.0

        if has_pose_raw and (pose_tracking_enabled or fall_state != STATE_IDLE) and current_state != STATE_SAFE_MODE:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

        if has_pose_raw:
            landmarks = results.pose_landmarks.landmark
            visible_x = []
            visible_y = []
            valid_keypoint_count = 0

            lower_body_idx = [25, 26, 27, 28]
            for idx in lower_body_idx:
                if landmarks[idx].visibility > POSE_KEYPOINT_VIS_THRESH:
                    lower_body_visible_count += 1

            for lm in landmarks:
                if lm.visibility > POSE_KEYPOINT_VIS_THRESH:
                    visible_x.append(lm.x)
                    visible_y.append(lm.y)
                    all_pose_points.append((lm.x * w, lm.y * h))
                    valid_keypoint_count += 1

            if valid_keypoint_count >= POSE_MIN_VALID_KEYPOINTS:
                body_w_norm = max(visible_x) - min(visible_x)
                body_h_norm = max(visible_y) - min(visible_y)
                body_aspect = body_w_norm / body_h_norm if body_h_norm > 0 else 999
                if 0.1 < body_aspect < 2.5:
                    pose_valid_this_frame = True
                    raw_center_x = (min(visible_x) + max(visible_x)) / 2
                    raw_center_y = (min(visible_y) + max(visible_y)) / 2
                    smooth_x, smooth_y = kf_pose.update(raw_center_x, raw_center_y)
                    current_pose_center = (smooth_x, smooth_y)

                    current_body_area = body_w_norm * body_h_norm
                    current_body_aspect = body_aspect

                    if current_pose_center is not None:
                        head_y = landmarks[0].y
                        head_y_history.append((time.time(), head_y))
                        if len(head_y_history) > 10: head_y_history.pop(0)

                    if (landmarks[11].visibility > 0.4 and landmarks[12].visibility > 0.4 and
                        landmarks[23].visibility > 0.4 and landmarks[24].visibility > 0.4):
                        shoulder_mid_y = (landmarks[11].y + landmarks[12].y) / 2
                        hip_mid_y = (landmarks[23].y + landmarks[24].y) / 2
                        torso_vertical = abs(hip_mid_y - shoulder_mid_y)
                        if body_h_norm > 0:
                            torso_ratio = torso_vertical / body_h_norm

                        shoulder_mid_x = (landmarks[11].x + landmarks[12].x) / 2
                        hip_mid_x = (landmarks[23].x + landmarks[24].x) / 2
                        dx = abs(hip_mid_x - shoulder_mid_x)
                        dy = abs(hip_mid_y - shoulder_mid_y)
                        torso_angle = np.degrees(np.arctan2(dx, dy)) if dy > 0 else 0.0

        if pose_valid_this_frame:
            pose_miss_count = 0
            pose_valid_streak += 1
            pose_predict_mode = False
            wander_predict_frames = 0
            pose_lost_frames = 0
        else:
            pose_miss_count += 1
            pose_valid_streak = 0

        has_pose = False
        if pose_valid_streak >= POSE_VALID_CONFIRM_FRAMES or pose_miss_count < POSE_MISS_TOLERANCE:
            has_pose = True
        elif POSE_PREDICT_ENABLE and pose_tracking_enabled:
            max_lost_frames = FALL_LOST_MAX_FRAMES if fall_state != STATE_IDLE else NORMAL_LOST_MAX_FRAMES
            if pose_lost_frames < max_lost_frames:
                pred_x, pred_y = kf_pose.predict()
                current_pose_center = (pred_x, pred_y)
                current_body_area = last_patient_body_area
                current_body_aspect = last_patient_body_aspect
                pose_predict_mode = True
                has_pose = True
                wander_predict_frames += 1
                pose_lost_frames += 1
                if pose_lost_frames > max_lost_frames // 3 and not fast_face_search:
                    fast_face_search = True
                    fast_search_frames = 0
            else:
                # 长时间丢失，退出追踪
                pose_tracking_enabled = False
                pose_was_lost = False
                pose_predict_mode = False
                has_patient_face = False
                last_patient_position = None
                last_patient_body_area = None
                last_patient_body_aspect = None
                last_known_patient_position = None
                last_known_body_area = None
                current_state = STATE_IDLE
                fall_state = STATE_IDLE
                fall_process_detected = False
                fall_process_last_time = 0.0
                post_fall_lock = False
                recovery_streak = 0
                stop_verify_buffer.clear()
                door_ratio_history.clear()
                wander_judged = False
                wander_printed = False
                wander_alert_active = False
                last_body_bbox = None
                person_switch_abnormal_count = 0
                motion_history.clear()
                kf_pose.reset()
                fast_face_search = False
                idle_printed = False
                pose_lost_printed = True
                wander_predict_frames = 0
                sim_smooth_buffer.clear()
                print("⚠️  骨骼长时间丢失，已退出追踪")
                continue

        person_switched = False
        in_fall_state = (fall_state != STATE_IDLE)

        if pose_tracking_enabled and has_pose and current_pose_center is not None and not in_fall_state and current_state != STATE_SAFE_MODE and not pose_predict_mode:
            motion_dist = 0.0
            if last_patient_position is not None:
                motion_dist = np.sqrt(
                    (current_pose_center[0] - last_patient_position[0])**2 +
                    (current_pose_center[1] - last_patient_position[1])**2
                )
            motion_history.append(motion_dist)

            avg_motion = np.mean(motion_history) if len(motion_history) > 0 else 0
            large_motion = avg_motion > 0.2

            abnormal = False
            if not large_motion:
                abnormal_count = 0
                if last_patient_position is not None and motion_dist > POSITION_JUMP_THRESHOLD:
                    abnormal_count += 1

                if last_patient_body_area is not None and current_body_area is not None and last_patient_body_area > 0:
                    area_change = abs(current_body_area - last_patient_body_area) / last_patient_body_area
                    if area_change > AREA_JUMP_THRESHOLD:
                        abnormal_count += 1

                if last_patient_body_aspect is not None and current_body_aspect is not None:
                    aspect_change = abs(current_body_aspect - last_patient_body_aspect)
                    if aspect_change > ASPECT_JUMP_THRESHOLD:
                        abnormal_count += 1

                if abnormal_count >= 2:
                    person_switch_abnormal_count += 1
                else:
                    person_switch_abnormal_count = max(0, person_switch_abnormal_count - 1)

                if person_switch_abnormal_count >= PERSON_SWITCH_CONFIRM_FRAMES:
                    person_switched = True
                    print("⚠️  检测到人体切换，已断开追踪")

            last_patient_position = current_pose_center
            last_patient_body_area = current_body_area
            last_patient_body_aspect = current_body_aspect

        if person_switched:
            pose_tracking_enabled = False
            pose_was_lost = True
            pose_predict_mode = False
            has_patient_face = False
            last_known_patient_position = last_patient_position
            last_known_body_area = last_patient_body_area
            last_patient_position = None
            last_patient_body_area = None
            last_patient_body_aspect = None
            current_state = STATE_IDLE
            fall_state = STATE_IDLE
            fall_process_detected = False
            fall_process_last_time = 0.0
            post_fall_lock = False
            recovery_streak = 0
            stop_verify_buffer.clear()
            door_ratio_history.clear()
            wander_judged = False
            wander_printed = False
            wander_alert_active = False
            last_body_bbox = None
            person_switch_abnormal_count = 0
            motion_history.clear()
            kf_pose.reset()
            fast_face_search = True
            fast_search_frames = 0
            idle_printed = False
            tracking_lost_printed = True
            wander_predict_frames = 0
            sim_smooth_buffer.clear()
            pose_lost_frames = 0
            fast_enter_streak = 0

        # 丢失后恢复追踪：强制人脸验证通过
        if pose_was_lost and has_pose and current_pose_center is not None and fast_face_search and not pose_predict_mode:
            pos_ok = last_known_patient_position is not None and np.sqrt(
                (current_pose_center[0] - last_known_patient_position[0])**2 +
                (current_pose_center[1] - last_known_patient_position[1])**2
            ) <= POSITION_RECOVER_THRESHOLD

            area_ok = False
            if last_known_body_area is not None and current_body_area is not None and last_known_body_area > 0:
                area_ok = abs(current_body_area - last_known_body_area) / last_known_body_area <= RECOVER_AREA_MAX_DIFF

            dynamic_verify_thresh = get_dynamic_verify_threshold(cached_patient_face_width)
            face_ok = cached_has_patient and cached_patient_similarity >= dynamic_verify_thresh

            if pos_ok and area_ok and face_ok:
                pose_tracking_enabled = True
                pose_was_lost = False
                pose_predict_mode = False
                has_patient_face = True
                fast_face_search = False
                last_patient_position = current_pose_center
                last_patient_body_area = current_body_area
                last_patient_body_aspect = current_body_aspect
                current_state = STATE_MONITORING
                person_switch_abnormal_count = 0
                motion_history.clear()
                pose_lost_frames = 0
                idle_printed = False
                monitor_printed = False
                wander_predict_frames = 0
                stop_verify_buffer.clear()
                sim_smooth_buffer.clear()
                kf_pose.reset()
                if current_pose_center is not None:
                    kf_pose.update(current_pose_center[0], current_pose_center[1])
                last_body_bbox = None
                print("✅ 位置+人脸校验通过，恢复追踪")
                if not patient_at_home:
                    patient_at_home = True
                    home_state_printed = False
                    print("🏠 患者重新出现，判定为在家")

        if pose_tracking_enabled and current_state == STATE_MONITORING:
            run_face_detect = (frame_counter % TRACK_DETECT_INTERVAL == 0)
        else:
            run_face_detect = (frame_counter % FACE_DETECTION_INTERVAL == 0)

        if fast_face_search and fast_search_frames < FAST_FACE_SEARCH_FRAMES:
            run_face_detect = True
            fast_search_frames += 1

        if run_face_detect:
            detections = face_engine.detect(frame)
            raw_faces = []

            for bbox, landmarks_5 in detections:
                loose_filter = pose_tracking_enabled or fast_face_search
                if not is_face_complete_by_bbox(bbox, w, h, loose_mode=loose_filter):
                    continue

                emb = face_engine.extract(frame, bbox, landmarks_5)
                if emb is None: continue
                sim, view = calc_view_matched_sim(emb, landmarks_5, feature_dict)
                raw_faces.append((bbox, sim, landmarks_5, emb, view))

            dedup = dedup_faces([(b, s) for b, s, _, _, _ in raw_faces])
            final_faces = []
            for (bbox, sim) in dedup:
                for orig in raw_faces:
                    if np.allclose(orig[0], bbox) and abs(orig[1] - sim) < 1e-6:
                        final_faces.append(orig)
                        break

            valid_face_count = 0
            for bbox, sim, landmarks_5, emb, view in final_faces:
                face_w = bbox[2] - bbox[0]
                completeness = calc_face_completeness(bbox, w, h)
                frontal_score = is_face_frontal(landmarks_5)
                yaw = estimate_yaw(landmarks_5)
                if face_w >= 40 and completeness >= 0.45 and frontal_score >= 0.35 and yaw < MAX_YAW_DEGREE:
                    valid_face_count += 1

            cached_total_face_count = len(final_faces)
            cached_valid_face_count = valid_face_count
            cached_has_patient = False
            cached_patient_face = None
            cached_patient_similarity = 0.0
            cached_patient_face_center_norm = None
            cached_patient_face_width = 0

            cached_nearest_face_exist = False
            cached_nearest_face_sim = 0.0
            cached_nearest_face_width = 0
            cached_nearest_face_bbox = None

            if pose_tracking_enabled and current_pose_center is not None and not person_switched:
                min_dist = float('inf')
                nearest_face = None
                nearest_sim = 0.0

                for bbox, sim, landmarks_5, emb, view in final_faces:
                    fc_x = (bbox[0] + bbox[2]) / 2 / w
                    fc_y = (bbox[1] + bbox[3]) / 2 / h
                    dist = np.sqrt(
                        (fc_x - current_pose_center[0])**2 +
                        (fc_y - current_pose_center[1])**2
                    )
                    if dist < FACE_POSE_MAX_DISTANCE and dist < min_dist:
                        min_dist = dist
                        nearest_face = bbox
                        nearest_sim = sim

                if nearest_face is not None:
                    cached_nearest_face_exist = True
                    cached_nearest_face_sim = nearest_sim
                    cached_nearest_face_width = nearest_face[2] - nearest_face[0]
                    cached_nearest_face_bbox = nearest_face

                    if cached_nearest_face_width >= 40:
                        dynamic_verify_thresh = get_dynamic_verify_threshold(cached_nearest_face_width)
                        if nearest_sim >= dynamic_verify_thresh:
                            cached_has_patient = True
                            cached_patient_face = nearest_face
                            cached_patient_similarity = nearest_sim
                            cached_patient_face_center_norm = (
                                (nearest_face[0]+nearest_face[2])/2/w,
                                (nearest_face[1]+nearest_face[3])/2/h
                            )
                            cached_patient_face_width = cached_nearest_face_width

            else:
                for bbox, sim, landmarks_5, emb, view in final_faces:
                    face_w = bbox[2] - bbox[0]
                    dynamic_enter_thresh = get_dynamic_enter_threshold(face_w)
                    if sim >= dynamic_enter_thresh:
                        if not cached_has_patient or sim > cached_patient_similarity:
                            cached_has_patient = True
                            cached_patient_face = bbox
                            cached_patient_similarity = sim
                            cached_patient_face_center_norm = (
                                (bbox[0]+bbox[2])/2/w,
                                (bbox[1]+bbox[3])/2/h
                            )
                            cached_patient_face_width = face_w

            if cached_has_patient:
                sim_smooth_buffer.append(cached_patient_similarity)
                cached_patient_similarity = float(np.mean(sim_smooth_buffer))
            else:
                sim_smooth_buffer.clear()

            # 进入追踪滑窗逻辑
            if current_state == STATE_IDLE and not fast_face_search and feature_dict:
                hit = False
                if cached_has_patient and current_pose_center is not None and cached_patient_face_center_norm is not None:
                    fd = np.sqrt(
                        (cached_patient_face_center_norm[0] - current_pose_center[0])**2 +
                        (cached_patient_face_center_norm[1] - current_pose_center[1])**2
                    )
                    if fd < FACE_POSE_MAX_DISTANCE:
                        hit = True
                enter_track_buffer.append(hit)

                if hit and pose_valid_streak >= POSE_ENTER_VALID_THRESH:
                    fast_enter_streak += 1
                else:
                    fast_enter_streak = 0

                enter_success = False
                if fast_enter_streak >= 3:
                    enter_success = True
                elif len(enter_track_buffer) >= ENTER_TRACK_WINDOW:
                    hit_count = sum(enter_track_buffer)
                    if hit_count >= ENTER_TRACK_MIN_HIT and pose_valid_streak >= POSE_ENTER_VALID_THRESH:
                        enter_success = True

                if enter_success:
                    pose_tracking_enabled = True
                    pose_was_lost = False
                    pose_predict_mode = False
                    has_patient_face = True
                    current_state = STATE_MONITORING
                    if current_pose_center is not None:
                        last_patient_position = current_pose_center
                        kf_pose.reset()
                        kf_pose.update(current_pose_center[0], current_pose_center[1])
                    last_patient_body_area = current_body_area
                    last_patient_body_aspect = current_body_aspect
                    stop_verify_buffer.clear()
                    person_switch_abnormal_count = 0
                    motion_history.clear()
                    door_ratio_history.clear()
                    wander_judged = False
                    wander_predict_frames = 0
                    sedentary_start_time = current_time
                    accumulated_sedentary_time = 0.0
                    last_sedentary_position = cached_patient_face_center_norm
                    last_ref_type = 'face'
                    last_sitting_state = True
                    sedentary_alert_display = False
                    sedentary_lost_frames = 0
                    current_sedentary_seconds = 0.0
                    sitting_streak = 0
                    last_raw_sitting = True
                    pose_lost_frames = 0
                    last_body_bbox = None
                    fast_enter_streak = 0
                    fall_process_detected = False
                    fall_process_last_time = 0.0
                    post_fall_lock = False
                    recovery_streak = 0
                    sedentary_disable_until_time = 0.0
                    if not patient_at_home:
                        patient_at_home = True
                        home_state_printed = False
                        print("🏠 检测到患者并锁定追踪，判定为在家")
                    if not monitor_printed:
                        print("✅ 锁定患者，进入监护追踪")
                        monitor_printed = True

            # 追踪中人脸周期验证
            if current_state == STATE_MONITORING and pose_tracking_enabled and feature_dict and not in_fall_state:
                if cached_nearest_face_exist and cached_nearest_face_bbox is not None:
                    target_face_width = cached_nearest_face_width
                    if target_face_width < 40:
                        pass
                    else:
                        target_completeness = calc_face_completeness(cached_nearest_face_bbox, w, h)
                        target_sim = cached_nearest_face_sim
                        if target_completeness >= FACE_COMPLETENESS_SKIP_THRESH:
                            dynamic_thresh = get_dynamic_verify_threshold(target_face_width)
                            is_fail = (target_sim < dynamic_thresh)
                            stop_verify_buffer.append(is_fail)

                            if len(stop_verify_buffer) >= VERIFY_WINDOW_SIZE:
                                fail_count = sum(stop_verify_buffer)
                                if fail_count >= VERIFY_FAIL_THRESHOLD:
                                    pose_tracking_enabled = False
                                    pose_was_lost = True
                                    pose_predict_mode = False
                                    has_patient_face = False
                                    last_known_patient_position = last_patient_position
                                    last_known_body_area = last_patient_body_area
                                    last_patient_position = None
                                    last_patient_body_area = None
                                    last_patient_body_aspect = None
                                    current_state = STATE_IDLE
                                    fall_state = STATE_IDLE
                                    fall_process_detected = False
                                    fall_process_last_time = 0.0
                                    post_fall_lock = False
                                    recovery_streak = 0
                                    door_ratio_history.clear()
                                    wander_judged = False
                                    wander_alert_active = False
                                    last_body_bbox = None
                                    person_switch_abnormal_count = 0
                                    motion_history.clear()
                                    stop_verify_buffer.clear()
                                    kf_pose.reset()
                                    fast_face_search = True
                                    fast_search_frames = 0
                                    idle_printed = False
                                    wander_predict_frames = 0
                                    sim_smooth_buffer.clear()
                                    pose_lost_frames = 0
                                    fast_enter_streak = 0
                                    print("⚠️  人脸验证连续失败，已退出追踪")

        # 人数统计
        if current_state == STATE_MONITORING and pose_tracking_enabled:
            display_face_count = max(cached_valid_face_count, 1)
        else:
            display_face_count = cached_valid_face_count

        has_patient_in_frame = cached_has_patient

        # ====================== 久坐检测（跌倒后强制禁用30秒）======================
        fall_disable_active = (current_time < last_fall_disable_end_time)
        sedentary_disable_active = (current_time < sedentary_disable_until_time)
        if current_state == STATE_MONITORING and pose_tracking_enabled and not fall_disable_active and not sedentary_disable_active and patient_at_home:
            current_ref_position = None
            current_ref_type = None

            if cached_has_patient and cached_patient_face_center_norm is not None:
                current_ref_position = cached_patient_face_center_norm
                current_ref_type = 'face'
                sedentary_lost_frames = 0
            elif has_pose and current_pose_center is not None:
                current_ref_position = current_pose_center
                current_ref_type = 'pose'
                sedentary_lost_frames = 0
            else:
                sedentary_lost_frames += 1

            sitting_raw = True
            if torso_ratio is not None and landmarks is not None:
                head_y = landmarks[0].y if landmarks[0].visibility > 0.5 else 1.0
                if lower_body_visible_count >= 2:
                    sitting_raw = (torso_ratio < SITTING_TORSO_RATIO_THRESH and head_y > SITTING_HEAD_Y_THRESH)
                else:
                    sitting_raw = (torso_ratio < SITTING_UPPER_BODY_RATIO_THRESH and head_y > SITTING_HEAD_Y_THRESH)

            if sitting_raw == last_raw_sitting:
                sitting_streak += 1
            else:
                sitting_streak = 1
                last_raw_sitting = sitting_raw

            if sitting_streak >= SITTING_CONFIRM_FRAMES:
                current_is_sitting = sitting_raw
            else:
                current_is_sitting = last_sitting_state if last_sitting_state is not None else True

            if current_ref_position is not None:
                if last_sedentary_position is None or sedentary_start_time is None:
                    sedentary_start_time = current_time
                    accumulated_sedentary_time = 0.0
                    last_sedentary_position = current_ref_position
                    last_ref_type = current_ref_type
                    last_sitting_state = current_is_sitting
                else:
                    if current_is_sitting != last_sitting_state:
                        sedentary_start_time = current_time
                        accumulated_sedentary_time = 0.0
                        last_sitting_state = current_is_sitting
                        sedentary_alert_display = False

                    ref_switched = (current_ref_type != last_ref_type)
                    move_dist = 0.0
                    if ref_switched:
                        last_sedentary_position = current_ref_position
                        last_ref_type = current_ref_type
                    else:
                        move_dist = np.sqrt(
                            (current_ref_position[0] - last_sedentary_position[0])**2 +
                            (current_ref_position[1] - last_sedentary_position[1])**2
                        )
                        if move_dist > SEDENTARY_MOVE_THRESHOLD:
                            sedentary_start_time = current_time
                            accumulated_sedentary_time = 0.0
                            last_sedentary_position = current_ref_position
                            sedentary_alert_display = False

                    if current_is_sitting:
                        current_sedentary_seconds = current_time - sedentary_start_time

                        if SEDENTARY_DEBUG and current_time - last_debug_print_time >= 1.0:
                            body_type = "全身" if lower_body_visible_count >=2 else "上半身"
                            status = f"坐姿({body_type})" if torso_ratio is not None else "坐姿(人脸)"
                            ratio_str = f"{torso_ratio:.3f}" if torso_ratio is not None else "--"
                            print(f"[久坐] {status} | 已坐 {current_sedentary_seconds:.1f}s | 位移 {move_dist:.4f} | 肩髋比 {ratio_str}")
                            last_debug_print_time = current_time

                        if (current_sedentary_seconds >= SEDENTARY_THRESHOLD_SECONDS and 
                            current_time - last_sedentary_alert_time > SEDENTARY_COOLDOWN_SECONDS):
                            last_sedentary_alert_time = current_time
                            sedentary_alert_display = True
                            sedentary_display_remain = 180
                            print(f"🪑 久坐提醒：患者已久坐超过 {SEDENTARY_THRESHOLD_SECONDS} 秒")
                    else:
                        current_sedentary_seconds = 0.0

                    last_sitting_state = current_is_sitting

            elif sedentary_lost_frames > SEDENTARY_LOST_MAX_FRAMES:
                sedentary_start_time = None
                accumulated_sedentary_time = 0.0
                last_sedentary_position = None
                last_ref_type = None
                last_sitting_state = None
                sedentary_alert_display = False
                sedentary_lost_frames = 0
                current_sedentary_seconds = 0.0
                sitting_streak = 0
                last_raw_sitting = True
        else:
            sedentary_start_time = None
            accumulated_sedentary_time = 0.0
            last_sedentary_position = None
            last_ref_type = None
            last_sitting_state = None
            sedentary_alert_display = False
            sedentary_lost_frames = 0
            current_sedentary_seconds = 0.0
            sitting_streak = 0
            last_raw_sitting = True

        # ====================== 跌倒检测======================
        if current_state == STATE_MONITORING or fall_state != STATE_IDLE:
            current_fall_condition = False
            current_recovery_condition = False
            aspect_ratio = 0.0
            head_y = 0.0
            body_height_val = 0.0
            head_speed = 0.0
            aspect_change = 0.0
            dynamic_process_triggered = False

            if not fall_disable_active and has_pose and results.pose_landmarks is not None and not pose_predict_mode:
                landmarks = results.pose_landmarks.landmark
                all_x = []
                all_y = []
                for lm in landmarks:
                    if lm.visibility > 0.5:
                        all_x.append(lm.x)
                        all_y.append(lm.y)

                if len(all_x) > 6:
                    min_x, max_x = min(all_x), max(all_x)
                    min_y, max_y = min(all_y), max(all_y)
                    width = max_x - min_x
                    height = max_y - min_y
                    aspect_ratio = width / height if height > 0 else 0
                    head_y = landmarks[0].y
                    body_height_val = height

                    if len(head_y_history) >= 3:
                        t1, y1 = head_y_history[0]
                        t2, y2 = head_y_history[-1]
                        time_diff = t2 - t1
                        if time_diff > 0.1:
                            head_speed = (y2 - y1) / time_diff

                    aspect_ratio_history.append((current_time, aspect_ratio))
                    if len(aspect_ratio_history) > HISTORY_FRAMES:
                        aspect_ratio_history.pop(0)

                    if len(aspect_ratio_history) >= 3:
                        t1, a1 = aspect_ratio_history[0]
                        t2, a2 = aspect_ratio_history[-1]
                        time_diff = t2 - t1
                        if time_diff > 0.1:
                            aspect_change = (a2 - a1) / time_diff

                    if (landmarks[11].visibility > 0.4 and landmarks[12].visibility > 0.4 and
                        landmarks[23].visibility > 0.4 and landmarks[24].visibility > 0.4):
                        shoulder_mid_x = (landmarks[11].x + landmarks[12].x) / 2
                        shoulder_mid_y = (landmarks[11].y + landmarks[12].y) / 2
                        hip_mid_x = (landmarks[23].x + landmarks[24].x) / 2
                        hip_mid_y = (landmarks[23].y + landmarks[24].y) / 2
                        dx = abs(hip_mid_x - shoulder_mid_x)
                        dy = abs(hip_mid_y - shoulder_mid_y)
                        torso_angle = np.degrees(np.arctan2(dx, dy)) if dy > 0 else 0.0

                    # 动态下落过程检测
                    if head_speed > FALL_PROCESS_HEAD_SPEED and abs(aspect_change) > FALL_PROCESS_ASPECT_CHANGE:
                        dynamic_process_triggered = True
                        fall_process_detected = True
                        fall_process_last_time = current_time

                    # 静态躺卧判定
                    strong_static = (
                        (aspect_ratio > FALL_STRONG_ASPECT and head_y > FALL_STRONG_HEAD_Y) or
                        torso_angle > FALL_STRONG_TORSO_ANGLE
                    )
                    weak_static = (
                        (aspect_ratio > FALL_WEAK_ASPECT and head_y > FALL_WEAK_HEAD_Y) or
                        torso_angle > FALL_WEAK_TORSO_ANGLE
                    )
                    dynamic_trigger = (
                        head_speed > FALL_HEAD_SPEED_THRESH or
                        abs(aspect_change) > FALL_ASPECT_CHANGE_THRESH
                    )

                    # 核心判定
                    process_in_valid_window = (current_time - fall_process_last_time) <= FALL_PROCESS_VALID_WINDOW
                    if fall_process_detected and process_in_valid_window and not post_fall_lock:
                        current_fall_condition = strong_static or (weak_static and dynamic_trigger) or (weak_static and (head_speed > FALL_PROCESS_HEAD_SPEED * 0.7))

                    # 恢复判定
                    current_recovery_condition = (
                        torso_angle < 20 and
                        aspect_ratio < FALL_RECOVERY_ASPECT_THRESH and
                        head_y < FALL_RECOVERY_HEAD_Y_THRESH
                    )
            
                    if current_recovery_condition and post_fall_lock:
                        recovery_streak += 1
                        if recovery_streak >= FALL_RECOVERY_CONFIRM_FRAMES:
                            post_fall_lock = False
                            fall_process_detected = False
                            recovery_streak = 0
                            print("✅ 姿态已恢复，解除跌倒锁定")
                    elif not current_recovery_condition:
                        recovery_streak = 0

            # 跌倒状态机
            if fall_state == STATE_IDLE and current_state == STATE_MONITORING and not fall_disable_active and not post_fall_lock:
                if current_fall_condition and (current_time - last_fall_detect_time) > FALL_COOLDOWN_SECONDS:
                    fall_count += 1
                    if fall_count >= FALL_CONFIRM_FRAMES:
                        fall_state = STATE_FALL_CONFIRM
                        final_confirm_start_time = current_time
                        fall_frames_in_confirm = 0
                        total_frames_in_confirm = 0
                        pose_lost_fall_count = 0
                        has_valid_fall_before_lost = False
                        fall_recovery_count = 0
                        consecutive_normal_frames = 0
                        fall_locked = False
                        last_fall_detect_time = current_time
                        sedentary_start_time = None
                        sedentary_disable_until_time = current_time + SEDENTARY_DISABLE_AFTER_FALL_SECONDS
                        print("🔍 检测到疑似跌倒，进入2秒确认阶段")
                else:
                    fall_count = max(0, fall_count - 1)

            elif fall_state == STATE_FALL_CONFIRM:
                total_frames_in_confirm += 1
                if fall_locked:
                    fall_state = STATE_FALL_ALERT
                    fall_alert_triggered = True
                    last_fall_alert_start_time = current_time
                    last_fall_disable_end_time = current_time + FALL_AFTER_DISABLE_SECONDS
                    sedentary_disable_until_time = current_time + SEDENTARY_DISABLE_AFTER_FALL_SECONDS
                    post_fall_lock = True
                    post_fall_lock_until = current_time + FALL_POST_ALERT_LOCK_SECONDS
                    print(f"🚨 跌倒状态锁定，确认报警，告警2秒后自动恢复，久坐禁用{SEDENTARY_DISABLE_AFTER_FALL_SECONDS}秒，姿态锁定{FALL_POST_ALERT_LOCK_SECONDS}秒")
                    continue

                if has_pose and results.pose_landmarks is not None and not pose_predict_mode:
                    pose_lost_fall_count = 0
                    if current_fall_condition:
                        fall_frames_in_confirm += 1
                        consecutive_normal_frames = 0
                        if fall_frames_in_confirm >= POSE_LOST_MIN_FALL_FRAMES:
                            has_valid_fall_before_lost = True
                        if fall_frames_in_confirm >= FALL_CONFIRM_IRREVERSIBLE_THRESHOLD:
                            fall_state = STATE_FALL_ALERT
                            fall_alert_triggered = True
                            last_fall_alert_start_time = current_time
                            last_fall_disable_end_time = current_time + FALL_AFTER_DISABLE_SECONDS
                            sedentary_disable_until_time = current_time + SEDENTARY_DISABLE_AFTER_FALL_SECONDS
                            post_fall_lock = True
                            post_fall_lock_until = current_time + FALL_POST_ALERT_LOCK_SECONDS
                            print(f"🚨 累计跌倒帧达标，确认患者跌倒，告警2秒后自动恢复，久坐禁用{SEDENTARY_DISABLE_AFTER_FALL_SECONDS}秒，姿态锁定{FALL_POST_ALERT_LOCK_SECONDS}秒")
                            continue
                    else:
                        consecutive_normal_frames += 1
                        if consecutive_normal_frames >= FALL_CONFIRM_RECOVERY_THRESHOLD:
                            fall_frames_in_confirm = max(0, fall_frames_in_confirm - 1)
                            consecutive_normal_frames = 0
                else:
                    pose_lost_fall_count += 1
                    consecutive_normal_frames = 0
                    if pose_lost_fall_count >= POSE_LOST_LOCK_FRAMES and not fall_locked:
                        fall_locked = True
                        continue
                    if pose_lost_fall_count >= POSE_LOST_FORCE_CONFIRM_FRAMES:
                        fall_state = STATE_FALL_ALERT
                        fall_alert_triggered = True
                        last_fall_alert_start_time = current_time
                        last_fall_disable_end_time = current_time + FALL_AFTER_DISABLE_SECONDS
                        sedentary_disable_until_time = current_time + SEDENTARY_DISABLE_AFTER_FALL_SECONDS
                        post_fall_lock = True
                        post_fall_lock_until = current_time + FALL_POST_ALERT_LOCK_SECONDS
                        print(f"🚨 骨骼持续丢失，强制确认跌倒，告警2秒后自动恢复，久坐禁用{SEDENTARY_DISABLE_AFTER_FALL_SECONDS}秒，姿态锁定{FALL_POST_ALERT_LOCK_SECONDS}秒")
                        continue
                    elif has_valid_fall_before_lost and pose_lost_fall_count >= POSE_LOST_CONFIRM_FRAMES:
                        fall_state = STATE_FALL_ALERT
                        fall_alert_triggered = True
                        last_fall_alert_start_time = current_time
                        last_fall_disable_end_time = current_time + FALL_AFTER_DISABLE_SECONDS
                        sedentary_disable_until_time = current_time + SEDENTARY_DISABLE_AFTER_FALL_SECONDS
                        post_fall_lock = True
                        post_fall_lock_until = current_time + FALL_POST_ALERT_LOCK_SECONDS
                        print(f"🚨 有效跌倒后骨骼丢失，自动确认跌倒，告警2秒后自动恢复，久坐禁用{SEDENTARY_DISABLE_AFTER_FALL_SECONDS}秒，姿态锁定{FALL_POST_ALERT_LOCK_SECONDS}秒")
                        continue

                if current_time - final_confirm_start_time > FINAL_CONFIRM_SECONDS:
                    if total_frames_in_confirm > 0 and fall_frames_in_confirm / total_frames_in_confirm >= FALL_RATIO_THRESHOLD:
                        fall_state = STATE_FALL_ALERT
                        fall_alert_triggered = True
                        last_fall_alert_start_time = current_time
                        last_fall_disable_end_time = current_time + FALL_AFTER_DISABLE_SECONDS
                        sedentary_disable_until_time = current_time + SEDENTARY_DISABLE_AFTER_FALL_SECONDS
                        post_fall_lock = True
                        post_fall_lock_until = current_time + FALL_POST_ALERT_LOCK_SECONDS
                        print(f"🚨 2秒确认超时，判定患者跌倒，告警2秒后自动恢复，久坐禁用{SEDENTARY_DISABLE_AFTER_FALL_SECONDS}秒，姿态锁定{FALL_POST_ALERT_LOCK_SECONDS}秒")
                    else:
                        fall_state = STATE_IDLE
                        fall_count = 0
                        pose_lost_fall_count = 0
                        has_valid_fall_before_lost = False
                        fall_recovery_count = 0
                        consecutive_normal_frames = 0
                        fall_locked = False
                        fall_process_detected = False
                        print("✅ 2秒确认期内恢复，取消跌倒预警")

            elif fall_state == STATE_FALL_ALERT:
                if current_time - last_fall_alert_start_time >= FALL_ALERT_DURATION_SECONDS:
                    fall_state = STATE_IDLE
                    fall_count = 0
                    pose_lost_fall_count = 0
                    has_valid_fall_before_lost = False
                    fall_recovery_count = 0
                    consecutive_normal_frames = 0
                    fall_locked = False
                    aspect_ratio_history.clear()
                    head_y_history.clear()
                    sedentary_start_time = current_time
                    print(f"✅ 跌倒告警已满{FALL_ALERT_DURATION_SECONDS}秒，自动恢复正常监护，姿态锁定剩余{max(0, post_fall_lock_until - current_time):.0f}秒")

            # 锁定超时自动解除
            if post_fall_lock and current_time > post_fall_lock_until:
                post_fall_lock = False
                fall_process_detected = False
                recovery_streak = 0
                print("⏱️  跌倒锁定超时，自动解除")

        # ====================== 门线走失检测 ======================
        if current_state == STATE_MONITORING or wander_alert_active:
            if has_pose and len(all_pose_points) > 0 and pose_tracking_enabled and not pose_predict_mode:
                xs = [p[0] for p in all_pose_points]
                ys = [p[1] for p in all_pose_points]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                bx1_raw = max(0, int(xmin - BODY_BOX_PADDING_X))
                bx2_raw = min(w, int(xmax + BODY_BOX_PADDING_X))
                by1_raw = max(0, int(ymin - BODY_BOX_PADDING_Y))
                by2_raw = min(h, int(ymax + BODY_BOX_PADDING_Y))

                if last_body_bbox is not None:
                    bx1 = int(last_body_bbox[0] * BOX_SMOOTH_ALPHA + bx1_raw * (1 - BOX_SMOOTH_ALPHA))
                    bx2 = int(last_body_bbox[1] * BOX_SMOOTH_ALPHA + bx2_raw * (1 - BOX_SMOOTH_ALPHA))
                    by1 = int(last_body_bbox[2] * BOX_SMOOTH_ALPHA + by1_raw * (1 - BOX_SMOOTH_ALPHA))
                    by2 = int(last_body_bbox[3] * BOX_SMOOTH_ALPHA + by2_raw * (1 - BOX_SMOOTH_ALPHA))
                else:
                    bx1, bx2, by1, by2 = bx1_raw, bx2_raw, by1_raw, by2_raw
                last_body_bbox = (bx1, bx2, by1, by2)

                last_body_door_ratio = calc_body_door_ratio(last_body_bbox, DOOR_POINTS)
                last_body_in_door = (last_body_door_ratio >= DOOR_PEAK_RATIO_THRESH)
                door_ratio_history.append(last_body_door_ratio)

                center_px = int((bx1 + bx2) / 2)
                center_py = int((by1 + by2) / 2)
                last_patient_center_pixel = (center_px, center_py)

            if pose_valid_streak >= POSE_VALID_CONFIRM_FRAMES and wander_judged:
                wander_judged = False
                wander_printed = False
                wander_alert_active = False

            trigger_wander = False
            if pose_was_lost and pose_lost_frames == WANDER_JUDGE_DELAY_FRAMES and not wander_judged:
                trigger_wander = True
            if pose_predict_mode and wander_predict_frames == WANDER_JUDGE_DELAY_FRAMES and not wander_judged:
                trigger_wander = True

            if trigger_wander:
                wander_judged = True
                if current_time - last_wander_alert_time > WANDER_COOLDOWN_SECONDS:
                    is_wander = False
                    peak_ratio = 0.0
                    last_3_avg = 0.0
                    center_in_door = False

                    if len(door_ratio_history) >= 5:
                        ratio_arr = np.array(door_ratio_history)
                        peak_ratio = np.max(ratio_arr)
                        last_3_avg = np.mean(ratio_arr[-3:])

                    if last_patient_center_pixel is not None:
                        center_in_door = is_point_in_door(last_patient_center_pixel, DOOR_POINTS)

                    is_wander = (peak_ratio >= DOOR_PEAK_RATIO_THRESH and last_3_avg >= DOOR_LAST_FRAMES_AVG_THRESH) or center_in_door
                    if is_wander:
                        wander_alert_active = True
                        wander_alert_remain = WANDER_ALERT_DISPLAY_FRAMES
                        wander_alert_triggered = True
                        last_wander_alert_time = current_time
                        if patient_at_home:
                            patient_at_home = False
                            home_state_printed = False
                            print("🚪 患者从门离开，判定为不在家")
                        if not wander_printed:
                            print(f"🚨 【走失告警】患者从门区域离开监控范围，告警自动显示{WANDER_ALERT_DISPLAY_FRAMES//15}秒")
                            wander_printed = True

        # ====================== 视频录制逻辑（含ntfy通知）======================
        if not is_recording and (fall_alert_triggered or wander_alert_triggered):
            if fall_alert_triggered:
                event_type = "fall"
                pre_seconds = FALL_PRE_RECORD_SECONDS
                post_seconds = FALL_POST_RECORD_SECONDS
                fall_alert_triggered = False
            else:
                event_type = "wander"
                pre_seconds = WANDER_PRE_RECORD_SECONDS
                post_seconds = WANDER_POST_RECORD_SECONDS
                wander_alert_triggered = False

            video_path = get_alert_video_path(event_type)
            pre_frames = []
            for t, buf_frame in frame_buffer:
                if current_time - t <= pre_seconds:
                    pre_frames.append(buf_frame)

            last_recorded_video_path = video_path
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

            for buf_frame in pre_frames:
                resized = cv2.resize(buf_frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
                video_writer.write(resized)

            is_recording = True
            recording_start_time = current_time
            recording_post_duration = post_seconds
            recording_event_type = event_type
            print(f"🎥 触发{event_type}告警录像，已保存到: {video_path}")

        if is_recording and video_writer is not None:
            resized_frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            video_writer.write(resized_frame)
            if current_time - recording_start_time >= recording_post_duration:
                video_writer.release()
                video_writer = None
                is_recording = False
                print(f"✅ {recording_event_type}现场录像录制完成")
                # 录制完成后发送ntfy通知
                if recording_event_type in ("fall", "wander", "manual"):
                    send_ntfy_alert(recording_event_type, last_recorded_video_path)

        # ====================== 多人安全模式 ======================
        safe_condition = (display_face_count >= 2)
        if safe_condition:
            safe_mode_enter_counter += 1
            safe_mode_exit_counter = 0
        else:
            safe_mode_enter_counter = 0
            safe_mode_exit_counter += 1

        if safe_mode_enter_counter >= SAFE_MODE_ENTER_FRAMES and current_state != STATE_SAFE_MODE:
            backup_tracking_enabled = pose_tracking_enabled
            backup_patient_position = last_patient_position
            backup_patient_body_area = last_patient_body_area
            backup_patient_body_aspect = last_patient_body_aspect
            backup_fall_state = fall_state

            current_state = STATE_SAFE_MODE
            stop_verify_buffer.clear()
            door_ratio_history.clear()
            wander_judged = False
            wander_printed = False
            enter_track_buffer.clear()
            fast_enter_streak = 0
            if not safe_mode_printed: 
                print("⚠️  检测到多人，进入安全模式，暂停主动监护")
                safe_mode_printed = True
        elif safe_mode_exit_counter >= SAFE_MODE_EXIT_FRAMES and current_state == STATE_SAFE_MODE:
            current_state = STATE_MONITORING if backup_tracking_enabled else STATE_IDLE
            pose_tracking_enabled = backup_tracking_enabled
            last_patient_position = backup_patient_position
            last_patient_body_area = backup_patient_body_area
            last_patient_body_aspect = backup_patient_body_aspect
            fall_state = backup_fall_state

            safe_mode_printed = False
            idle_printed = False
            monitor_printed = False
            wander_judged = False
            wander_printed = False
            print("✅ 人员减少，退出安全模式，已恢复监护状态")

        # ====================== 非安全模式主状态更新 ======================
        if current_state != STATE_SAFE_MODE:
            if has_patient_in_frame and cached_patient_face is not None:
                patient_bbox = cached_patient_face
                patient_sim = cached_patient_similarity
                face_detected = True
            else:
                face_detected = False

            if current_state == STATE_IDLE and not pose_tracking_enabled:
                if not idle_printed: 
                    print("❓ 空闲状态，等待患者出现")
                    idle_printed = True

        # ====================== UI绘制 ======================
        cv2.polylines(frame, [DOOR_POINTS], True, DOOR_LINE_COLOR, 2)
        door_label_x = DOOR_POINTS[0][0] + 5
        door_label_y = max(5, DOOR_POINTS[0][1] - 22)
        frame = put_chinese_text(frame, "门区域", (door_label_x, door_label_y), DOOR_LINE_COLOR, 14)

        if _calibrating_door and len(_calibration_points) > 0:
            for pt in _calibration_points:
                cv2.circle(frame, tuple(pt), 5, (0, 0, 255), -1)
            if len(_calibration_points) >= 2:
                pts_arr = np.array(_calibration_points, dtype=np.int32)
                cv2.polylines(frame, [pts_arr], False, (0, 0, 255), 2)
            frame = put_chinese_text(frame, f"门线标定中: {len(_calibration_points)}/4 点", (15, h - 30), (0, 0, 255), 16)

        # 左上角叠加信息
        y_offset = 20
        home_text = "🏠 患者在家" if patient_at_home else "🚪 患者不在家"
        home_color = (0, 255, 0) if patient_at_home else (0, 0, 255)
        frame = put_chinese_text(frame, home_text, (15, y_offset), home_color, font_size=FONT_SIZE+2)
        y_offset += 32

        if current_state != STATE_SAFE_MODE:
            state_text = {STATE_IDLE: "❓ 空闲", STATE_MONITORING: "✅ 监护中"}.get(current_state, "")
            if state_text: 
                frame = put_chinese_text(frame, state_text, (15, y_offset), (0, 255, 0))
                y_offset += 28

            frame = put_chinese_text(frame, f"👥 人数: {display_face_count}", (15, y_offset), (255, 255, 0))
            y_offset += 28

            if current_state == STATE_MONITORING and last_body_bbox is not None:
                frame = put_chinese_text(frame, f"门内占比: {last_body_door_ratio:.1%}", (15, y_offset), (0, 165, 255))
                y_offset += 28
                if pose_predict_mode:
                    frame = put_chinese_text(frame, "🔮 预测续追中", (15, y_offset), (255, 165, 0))
                    y_offset += 28

            # 坐姿显示（禁用期间不显示）
            if current_state == STATE_MONITORING and last_sitting_state and not sedentary_disable_active and patient_at_home:
                if sedentary_alert_display:
                    sed_text = "🪑 久坐告警！"
                    sed_color = (0, 0, 255)
                else:
                    remain = max(0, SEDENTARY_THRESHOLD_SECONDS - current_sedentary_seconds)
                    if torso_ratio is not None:
                        body_tag = "全身" if lower_body_visible_count >=2 else "上半身"
                        sed_text = f"🪑 坐姿({body_tag}) | 倒计时: {remain:.0f}s"
                    else:
                        sed_text = f"🪑 坐姿(人脸) | 倒计时: {remain:.0f}s"
                    sed_color = (0, 255, 255) if remain > 10 else (0, 165, 255)
                frame = put_chinese_text(frame, sed_text, (15, y_offset), sed_color)
                y_offset += 28

            if fall_state == STATE_FALL_CONFIRM:
                frame = put_chinese_text(frame, "🔍 跌倒确认中", (15, y_offset), (0, 165, 255))
                y_offset += 28

            if fall_state == STATE_FALL_ALERT:
                remain_time = max(0, FALL_ALERT_DURATION_SECONDS - (current_time - last_fall_alert_start_time))
                cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 255), -1)
                frame = put_chinese_text(frame, "🚨 患者跌倒！", (15, 20), (255, 255, 255), 24)
                frame = put_chinese_text(frame, f"告警剩余 {remain_time:.1f}秒后自动恢复", (15, 55), (255, 255, 255), 16)

            if wander_alert_active:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                frame = put_chinese_text(frame, "🚨 患者走失告警", (w//2 - 130, h//2 - 40), (0, 0, 255), 32)
                frame = put_chinese_text(frame, "患者从门区域离开监控范围", (w//2 - 150, h//2 + 20), (255, 255, 255), 20)
                wander_alert_remain -= 1
                if wander_alert_remain <= 0:
                    wander_alert_active = False

            if is_recording:
                elapsed = current_time - recording_start_time
                remaining = recording_post_duration - elapsed
                frame = put_chinese_text(frame, f"🔴 录制中: {remaining:.1f}s", (w - 220, 20), (0, 0, 255))

            # 身体框绘制
            if pose_tracking_enabled and last_body_bbox is not None:
                bx1, bx2, by1, by2 = last_body_bbox
                if last_body_in_door:
                    box_color = (0, 0, 255)
                    label = "患者 | 门内区域"
                elif pose_predict_mode:
                    box_color = (255, 165, 0)
                    label = "患者 | 预测续追"
                else:
                    box_color = (0, 255, 255)
                    label = "患者 | 骨骼追踪"
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2)
                frame = put_chinese_text(frame, label, (bx1, max(0, by1 - 25)), box_color, 16)

            # 人脸框绘制
            if face_detected and patient_bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in patient_bbox]
                face_w_px = int(x2 - x1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frame = put_chinese_text(frame, f"宽:{face_w_px}px | 相似度:{patient_sim:.2f}", 
                                        (x1, max(0, y1 - 45)), (0, 255, 0), 14)
        else:
            cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 255), -1)
            frame = put_chinese_text(frame, "⚠️ 多人安全模式", (w//2-100, 20), (255, 255, 255), 24)
            frame = put_chinese_text(frame, "暂关闭主动监护功能", (w//2-80, 50), (255, 255, 255), 16)

        # 直接显示原始画面（全屏由fullscreen_resize自动处理）
        cv2.imshow(WINDOW_NAME, fullscreen_resize(frame, WINDOW_NAME))

    # 资源释放
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    face_engine.release()
    if video_writer is not None:
        video_writer.release()
    # 清理 cloudflared 进程
    if cloudflared_proc is not None:
        cloudflared_proc.terminate()
        try:
            cloudflared_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            cloudflared_proc.kill()
        print("✅ cloudflared 隧道已关闭")
    print("\n✅ 程序已正常退出")


if __name__ == "__main__":
    main()
