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


# ---- Rectification (A3, A4, H4) & residual measurements ----
def perform_rectifications(img: np.ndarray, corners: np.ndarray):
    src = corners.astype(np.float32)  # [TL, TR, BR, BL]

    # Target points for a 450x450 square
    dst_pts = np.float32([[0, 0], [RECT_SIZE - 1, 0], [RECT_SIZE - 1, RECT_SIZE - 1], [0, RECT_SIZE - 1]])  # pyright: ignore[reportArgumentType]

    # A3: 3 point Affine (using TL, TR, BL)
    M_A3 = cv2.getAffineTransform(src[[0, 1, 3]], dst_pts[[0, 1, 3]])  # pyright: ignore[reportCallIssue, reportArgumentType]
    rect_A3 = cv2.warpAffine(img, M_A3, (RECT_SIZE, RECT_SIZE))

    # Calculate where original BR corner lands in A3 target space
    br_src_homo = np.array([src[2][0], src[2][1], 1.0], dtype=np.float32)
    br_mapped = M_A3 @ br_src_homo
    a3_corner_err_px = float(np.linalg.norm(br_mapped - dst_pts[2]))  # pyright: ignore[reportCallIssue, reportArgumentType]

    # A4: 4 point Least-Squares Affine
    M_A4, _ = cv2.estimateAffine2D(src, dst_pts)  # pyright: ignore[reportCallIssue, reportArgumentType]
    rect_A4 = cv2.warpAffine(img, M_A4, (RECT_SIZE, RECT_SIZE))

    # Calculate RMS residual over all 4 corners for A4
    src_homo = np.hstack([src, np.ones((4, 1), dtype=np.float32)])  # 4x3
    mapped_A4 = (M_A4 @ src_homo.T).T  # 4x2
    a4_rms_px = float(np.sqrt(np.mean(np.sum((mapped_A4 - dst_pts) ** 2, axis=1))))

    # H4: 4 point Projective Homography
    M_H4 = cv2.getPerspectiveTransform(src, dst_pts)  # pyright: ignore[reportCallIssue, reportArgumentType]
    rect_H4 = cv2.warpPerspective(img, M_H4, (RECT_SIZE, RECT_SIZE))

    return rect_A3, rect_A4, rect_H4, M_A3, M_A4, M_H4, a3_corner_err_px, a4_rms_px


# ---- Cell extraction and digit OCR ----
def is_cell_empty(cell_gray: np.ndarray):
    h, w = cell_gray.shape
    
    # Isolate the central core to avoid outer grid lines
    y1, y2 = int(h * 0.20), int(h * 0.80)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    core = cell_gray[y1:y2, x1:x2]

    # Contrast / Standard Deviation Check
    # Blank paper has low standard deviation; printed digits create high contrast.
    std_dev = float(np.std(core))
    ptp_range = float(np.ptp(core))  # Max minus min pixel intensity

    if std_dev < 12.0 or ptp_range < 40:
        return True  # Uniform region = Empty Paper

    # Relative Dark Pixel Count
    # Measure pixels significantly darker than the local average paper color of this cell
    mean_val = float(np.mean(core))
    dark_pixels = np.sum(core < (mean_val - 35))
    dark_ratio = dark_pixels / float(core.size)

    # If less than 3.5% of core pixels are dark ink, deem empty
    if dark_ratio < 0.035:
        return True

    return False