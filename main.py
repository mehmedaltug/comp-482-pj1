import os
import sys
import json
import time
import glob
import argparse
import cv2
import numpy as np
import pytesseract as pyt

RECT_SIZE = 450
CELL_SIZE = 50

# ---- Preprocessing (Scale Invariant) -----
def preprocess_image(img: np.ndarray):
    h, w = img.shape[:2]
    max_dim = max(h, w)
    scale = 1.0
    
    # Process on a normalized dimension for speed & consistency
    target_max_dim = 1200
    if max_dim > target_max_dim:
        scale = target_max_dim / float(max_dim)
        proc_w, proc_h = int(w * scale), int(h * scale)
        img_proc = cv2.resize(img, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    else:
        img_proc = img.copy()

    gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
    
    # Kernel sizes scaled relative to image size
    k_blur = max(3, int(round(img_proc.shape[1] * 0.005)) | 1)
    blur = cv2.GaussianBlur(gray, (k_blur, k_blur), 0)
    
    # Adaptive thresholding to handle lighting variations
    block_size = max(11, int(round(img_proc.shape[1] * 0.015)) | 1)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 2
    )
    
    return gray, thresh, scale

