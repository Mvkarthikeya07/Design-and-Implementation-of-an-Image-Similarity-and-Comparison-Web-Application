# utils/image_utils.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Tuple
from sklearn.cluster import KMeans

def read_and_limit_image(path: str, max_width: int = 1200):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def compute_ssim_vis(imgA_color, imgB_color) -> Tuple[float, np.ndarray]:
    """
    Returns (ssim_score, visualization_bgr)
    visualization is a colored heatmap (BGR uint8)
    """
    A_gray = to_gray(imgA_color)
    B_gray = to_gray(imgB_color)
    if A_gray.shape != B_gray.shape:
        B_gray = cv2.resize(B_gray, (A_gray.shape[1], A_gray.shape[0]), interpolation=cv2.INTER_AREA)
        imgB_color_resized = cv2.resize(imgB_color, (A_gray.shape[1], A_gray.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        imgB_color_resized = imgB_color

    score, diff = ssim(A_gray, B_gray, full=True)
    # Normalize diff to 0..255
    diff_norm = (255 * (diff - diff.min()) / max(1e-8, (diff.max() - diff.min()))).astype("uint8")
    diff_color = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    # Overlay diff on top of average image for nicer visualization
    avg = cv2.addWeighted(imgA_color, 0.6, imgB_color_resized, 0.4, 0)
    overlay = cv2.addWeighted(avg, 0.7, diff_color, 0.3, 0)
    return float(score), overlay

def orb_match_and_visualize(imgA_color, imgB_color, nfeatures=1000, ratio_thresh=0.75):
    """
    Returns (match_score, good_match_count, vis_bgr)
    vis_bgr: a visualization image (BGR) with matches drawn or side-by-side fallback.
    """
    A_gray = to_gray(imgA_color)
    B_gray = to_gray(imgB_color)
    if A_gray.shape != B_gray.shape:
        B_gray = cv2.resize(B_gray, (A_gray.shape[1], A_gray.shape[0]), interpolation=cv2.INTER_AREA)
        imgB_color = cv2.resize(imgB_color, (A_gray.shape[1], A_gray.shape[0]), interpolation=cv2.INTER_AREA)

    orb = cv2.ORB_create(nfeatures)
    kp1, des1 = orb.detectAndCompute(A_gray, None)
    kp2, des2 = orb.detectAndCompute(B_gray, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        # fallback side-by-side
        h = max(imgA_color.shape[0], imgB_color.shape[0])
        canvas = np.zeros((h, imgA_color.shape[1] + imgB_color.shape[1], 3), dtype=np.uint8)
        canvas[:imgA_color.shape[0], :imgA_color.shape[1]] = imgA_color
        canvas[:imgB_color.shape[0], imgA_color.shape[1]:] = imgB_color
        cv2.putText(canvas, "No ORB descriptors found", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        return 0.0, 0, canvas

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for m_n in knn:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    denom = max(1, min(len(kp1), len(kp2)))
    score = len(good) / denom
    # draw matches
    vis = cv2.drawMatches(imgA_color, kp1, imgB_color, kp2, good, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return float(score), len(good), vis

def combine_scores(ssim_score, orb_score, w_ssim=0.6, w_orb=0.4):
    total = w_ssim + w_orb
    w_ssim /= total; w_orb /= total
    combined = w_ssim * float(ssim_score) + w_orb * float(orb_score)
    return float(combined)
