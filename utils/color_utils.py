# utils/color_utils.py
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

# Basic named color palette (RGB)
COLOR_PALETTE = {
    'black': (0,0,0), 'white': (255,255,255), 'red': (220,20,60), 'green': (34,139,34),
    'blue': (30,144,255), 'yellow': (255,215,0), 'cyan': (0,206,209), 'magenta': (255,105,180),
    'gray': (128,128,128), 'orange': (255,140,0), 'brown': (160,82,45), 'pink': (255,182,193),
    'olive': (128,128,0), 'purple': (148,0,211)
}

def _closest_color_name(rgb):
    r,g,b = rgb
    best = None
    best_dist = float("inf")
    for name, val in COLOR_PALETTE.items():
        vr, vg, vb = val
        d = (r-vr)**2 + (g-vg)**2 + (b-vb)**2
        if d < best_dist:
            best_dist = d
            best = name
    return best

def dominant_color_names(img_bgr, k=3, sample_size=50000):
    """
    Returns list of k color names ordered by dominance.
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pixels = img.reshape(-1, 3).astype(float)
    if len(pixels) > sample_size:
        idx = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[idx]
    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(pixels)
    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = Counter(labels)
    ordered = [c for c, _ in sorted(counts.items(), key=lambda x: -x[1])]
    color_names = []
    for idx in ordered:
        rgb = tuple(centers[idx])
        color_names.append(_closest_color_name(rgb))
    # dedupe preserve order
    seen = set(); out=[]
    for c in color_names:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def common_color_names(listA, listB):
    return [c for c in listA if c in listB]
