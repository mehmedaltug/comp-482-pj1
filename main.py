from typing import Any, Optional, Dict, List, Tuple
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


# ---- Harris corner detection & automatic corner selection ----
def order_corners(pts: np.ndarray):
    pts = pts.reshape((4, 2)).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Top left has minimum (x + y), bottom right has maximum (x + y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top right has minimum (y - x), bottom left has maximum (y - x)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def detect_harris_corners(gray: np.ndarray, thresh: np.ndarray):

    # Compute Harris Response Map
    gray_f32 = np.float32(gray)
    block_size = max(2, int(round(gray.shape[1] * 0.003)))
    harris_resp = cv2.cornerHarris(gray_f32, blockSize=block_size, ksize=3, k=0.04)  # pyright: ignore[reportCallIssue, reportArgumentType]
    
    # Normalize Harris map for visualization & thresholding
    harris_norm = cv2.normalize(harris_resp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # pyright: ignore[reportCallIssue, reportArgumentType]

    # Extract Candidate Points via Non Maximum Suppression
    resp_thresh = 0.01 * harris_resp.max()
    harris_thresh = harris_resp > resp_thresh

    # Perform NMS using dilation
    dilated = cv2.dilate(harris_resp, None)  # pyright: ignore[reportCallIssue, reportArgumentType]
    nms_mask = (harris_resp == dilated) & harris_thresh

    candidate_coords = np.argwhere(nms_mask)  # returns (y, x)
    candidates = candidate_coords[:, [1, 0]].astype(np.float32)  # convert to (x, y)

    if len(candidates) < 4:
        return None, harris_norm, candidates

    # Restrict Search Area to the Largest Quadrilateral Region
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, harris_norm, candidates

    max_contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)

    if len(approx) == 4:
        rough_quad = approx.reshape(4, 2)
    else:
        hull = cv2.convexHull(max_contour).reshape(-1, 2)
        s = hull.sum(axis=1)
        diff = np.diff(hull, axis=1)
        rough_quad = np.array([hull[np.argmin(s)], hull[np.argmin(diff)], hull[np.argmax(s)], hull[np.argmax(diff)]])

    rough_quad = order_corners(rough_quad)

    # Select the closest Harris candidate for each corner of the rough region
    selected_corners = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        target_pt = rough_quad[i]
        dists = np.linalg.norm(candidates - target_pt, axis=1)
        best_idx = np.argmin(dists)
        selected_corners[i] = candidates[best_idx]

    # Refine selected Harris corners with sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    corners_subpix = cv2.cornerSubPix(
        gray, selected_corners.copy(), winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria
    )

    return order_corners(corners_subpix), harris_norm, candidates

def draw_harris_debug(harris_norm: np.ndarray, candidates: np.ndarray, corners: Optional[np.ndarray]):
    vis = cv2.cvtColor(harris_norm, cv2.COLOR_GRAY2BGR)

    # Plot candidate points (blue circles)
    for pt in candidates:
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (255, 120, 0), -1)

    # Highlight final 4 Harris corners (red circles with labels)
    if corners is not None:
        labels = ["TL", "TR", "BR", "BL"]
        colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 255)]
        for pt, label, col in zip(corners, labels, colors):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(vis, (x, y), 6, col, -1)
            cv2.circle(vis, (x, y), 8, (255, 255, 255), 2)
            cv2.putText(vis, label, (x + 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    return vis