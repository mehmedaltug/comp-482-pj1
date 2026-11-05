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

def preprocess_cell_for_ocr(cell_gray: np.ndarray):
    h, w = cell_gray.shape
    margin = int(h * 0.15)
    crop = cell_gray[margin : h - margin, margin : w - margin]

    # Relative binarization against local average paper intensity
    mean_val = float(np.mean(crop))
    binary = np.where(crop < (mean_val - 30), 255, 0).astype(np.uint8)

    # Find candidate digit contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Keep largest connected component
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    x, y, bw, bh = cv2.boundingRect(c)

    # Noise filter: minimum size thresholds for printed digits
    if area < 25 or bh < 10 or bw < 4:
        return None

    digit_crop = binary[y : y + bh, x : x + bw]

    # Invert binary mask so digit is black on white paper background
    digit_black_text = cv2.bitwise_not(digit_crop)

    # Center digit inside a 48x48 canvas
    padded = np.full((48, 48), 255, dtype=np.uint8)
    sx = (48 - bw) // 2
    sy = (48 - bh) // 2
    padded[sy : sy + bh, sx : sx + bw] = digit_black_text

    return padded

def extract_grid_and_ocr(rect_H4: np.ndarray):
    gray = cv2.cvtColor(rect_H4, cv2.COLOR_BGR2GRAY) if len(rect_H4.shape) == 3 else rect_H4
    
    grid = []
    ink_ratios = []
    tess_config = "--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"

    for r in range(9):
        row_digits = []
        row_ratios = []
        for c in range(9):
            y1, y2 = r * CELL_SIZE, (r + 1) * CELL_SIZE
            x1, x2 = c * CELL_SIZE, (c + 1) * CELL_SIZE
            cell_gray = gray[y1:y2, x1:x2]

            # EXPLICIT PRE-OCR EMPTY CHECK
            if is_cell_empty(cell_gray):
                row_digits.append(0)
                row_ratios.append(0.0)
            else:
                # PREPARE CELL & RUN OCR
                cell_ocr_input = preprocess_cell_for_ocr(cell_gray)
                if cell_ocr_input is None:
                    row_digits.append(0)
                    row_ratios.append(0.0)
                else:
                    try:
                        text = pyt.image_to_string(cell_ocr_input, config=tess_config).strip()
                        digit = int(text) if text.isdigit() and len(text) == 1 else 0
                    except Exception:
                        digit = 0
                    
                    row_digits.append(digit)
                    row_ratios.append(1.0)

        grid.append(row_digits)
        ink_ratios.append(row_ratios)

    return grid, ink_ratios


# ---- Validity check and visual debug rendering ----
def validate_sudoku_grid(grid: List[List[int]]):
    conflicts = set()

    # Check Rows
    for r in range(9):
        seen = {}
        for c in range(9):
            val = grid[r][c]
            if val != 0:
                if val in seen:
                    conflicts.add((r, c))
                    conflicts.add((r, seen[val]))
                else:
                    seen[val] = c

    # Check Columns
    for c in range(9):
        seen = {}
        for r in range(9):
            val = grid[r][c]
            if val != 0:
                if val in seen:
                    conflicts.add((r, c))
                    conflicts.add((seen[val], c))
                else:
                    seen[val] = r

    # Check 3x3 Boxes
    for box_row in range(3):
        for box_col in range(3):
            seen = {}
            for r in range(box_row * 3, (box_row + 1) * 3):
                for c in range(box_col * 3, (box_col + 1) * 3):
                    val = grid[r][c]
                    if val != 0:
                        if val in seen:
                            conflicts.add((r, c))
                            conflicts.add(seen[val])
                        else:
                            seen[val] = (r, c)

    verdict = "valid" if len(conflicts) == 0 else "invalid"
    return verdict, sorted(list(conflicts))

def render_debug_image(rect_H4: np.ndarray, grid: List[List[int]], conflicts: List[Tuple[int, int]], verdict: str):
    debug_vis = rect_H4.copy()

    # Draw 9x9 Grid Lines
    for i in range(10):
        pos = i * CELL_SIZE
        thick = 3 if i % 3 == 0 else 1
        color = (0, 0, 0)
        cv2.line(debug_vis, (pos, 0), (pos, RECT_SIZE), color, thick)
        cv2.line(debug_vis, (0, pos), (RECT_SIZE, pos), color, thick)

    conflict_set = set(conflicts)

    # Render Overlay Text & Conflict Boxes
    for r in range(9):
        for c in range(9):
            val = grid[r][c]
            x1, y1 = c * CELL_SIZE, r * CELL_SIZE
            x2, y2 = (c + 1) * CELL_SIZE, (r + 1) * CELL_SIZE

            is_conflict = (r, c) in conflict_set
            if is_conflict:
                # Fill conflicting cell with red tint overlay
                overlay = debug_vis.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.35, debug_vis, 0.65, 0, debug_vis)
                cv2.rectangle(debug_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if val != 0:
                txt_color = (0, 0, 220) if is_conflict else (200, 30, 0)
                # Center text inside cell
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.9
                thick = 2
                (tw, th), _ = cv2.getTextSize(str(val), font, scale, thick)
                tx = x1 + (CELL_SIZE - tw) // 2
                ty = y1 + (CELL_SIZE + th) // 2
                cv2.putText(debug_vis, str(val), (tx, ty), font, scale, txt_color, thick)

    # Header status banner
    banner_color = (0, 150, 0) if verdict == "valid" else (0, 0, 200)
    cv2.putText(
        debug_vis, f"Verdict: {verdict.upper()}", (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, banner_color, 2
    )

    return debug_vis


# ---- Execution for single image ----
def process_single_image(image_path: str, out_dir: str) -> Dict[str, Any]:
    start_time = time.time()
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return {"image": os.path.basename(image_path), "verdict": "not_found", "runtime_s": 0.0}

    # Preprocessing
    gray, thresh, scale = preprocess_image(img)

    # Harris Corner Detection & Corner Selection
    corners_proc, harris_norm, candidates = detect_harris_corners(gray, thresh)

    # Save <name>_harris.png
    harris_vis = draw_harris_debug(harris_norm, candidates, corners_proc)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_harris.png"), harris_vis)

    if corners_proc is None:
        print(f"Could not isolate Sudoku corners in {image_path}")
        result_data = {
            "image": os.path.basename(image_path),
            "corners": [],
            "A3": [],
            "A4": [],
            "H4": [],
            "a3_corner_err_px": None,
            "a4_rms_px": None,
            "grid": [],
            "verdict": "not_found",
            "runtime_s": round(time.time() - start_time, 4)
        }
        with open(os.path.join(out_dir, f"{base_name}_result.json"), "w") as f:
            json.dump(result_data, f, indent=2)
        return result_data

    # Map corner coordinates back to original unscaled image resolution
    corners_orig = corners_proc / scale

    # Rectification (A3, A4, H4)
    rect_A3, rect_A4, rect_H4, M_A3, M_A4, M_H4, a3_err, a4_rms = perform_rectifications(
        img, corners_orig
    )

    # Save <name>_rect_A3.png, _rect_A4.png, _rect_H4.png
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_rect_A3.png"), rect_A3)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_rect_A4.png"), rect_A4)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_rect_H4.png"), rect_H4)

    # Cell Extraction & Digit OCR (on H4 image)
    grid, _ = extract_grid_and_ocr(rect_H4)

    # Validity Verification
    verdict, conflicts = validate_sudoku_grid(grid)

    # Save <name>_grid.txt
    grid_txt_path = os.path.join(out_dir, f"{base_name}_grid.txt")
    with open(grid_txt_path, "w") as f:
        for row in grid:
            f.write("".join(map(str, row)) + "\n")

    # Save <name>_debug.png
    debug_vis = render_debug_image(rect_H4, grid, conflicts, verdict)
    cv2.imwrite(os.path.join(out_dir, f"{base_name}_debug.png"), debug_vis)

    runtime_s = round(time.time() - start_time, 4)

    # Format result dictionary matching Appendix A schema
    result_data = {
        "image": os.path.basename(image_path),
        "corners": corners_orig.tolist(),
        "A3": M_A3.tolist(),
        "A4": M_A4.tolist(),
        "H4": M_H4.tolist(),
        "a3_corner_err_px": round(a3_err, 4),
        "a4_rms_px": round(a4_rms, 4),
        "grid": grid,
        "verdict": verdict,
        "conflicting_cells": conflicts,
        "runtime_s": runtime_s
    }

    # Save <name>_result.json
    with open(os.path.join(out_dir, f"{base_name}_result.json"), "w") as f:
        json.dump(result_data, f, indent=2)

    print(f"Processed {image_path} in {runtime_s:.2f}s | Verdict: {verdict}")
    return result_data


# ---- Main function ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs="?", help="Path to single input image file")
    parser.add_argument("--batch", type=str, help="Path to directory containing input images (batch mode)")
    parser.add_argument("--outdir", "-o", type=str, default="out/", help="Directory to save output files")

    args = parser.parse_args()

    if not args.input and not args.batch:
        parser.print_help()
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    if args.batch:
        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        image_files = []
        for ext in valid_exts:
            image_files.extend(glob.glob(os.path.join(args.batch, ext)))
            image_files.extend(glob.glob(os.path.join(args.batch, ext.upper())))

        image_files = sorted(list(set(image_files)))
        print(f"--- Running Batch Mode on {len(image_files)} images ---")
        for img_path in image_files:
            process_single_image(img_path, args.outdir)
    else:
        process_single_image(args.input, args.outdir)

if __name__ == "__main__":
    main()