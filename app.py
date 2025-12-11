# app.py
import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from utils.image_utils import (
    read_and_limit_image, to_gray, compute_ssim_vis,
    orb_match_and_visualize, combine_scores
)
from utils.color_utils import dominant_color_names, common_color_names

# --- Configuration ---
UPLOAD_FOLDER = "static/outputs"
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}
MAX_WIDTH = 1200  # resize width to keep processing fast

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace_this_with_a_random_secret_in_production"

def allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower()
    return "." in filename and ext in ALLOWED_EXT

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare():
    # Accept two uploaded images
    file1 = request.files.get("image1")
    file2 = request.files.get("image2")
    if not file1 or not file2:
        flash("Please upload both images.")
        return redirect(url_for("index"))
    if not (allowed_file(file1.filename) and allowed_file(file2.filename)):
        flash("Allowed file types: png, jpg, jpeg, bmp")
        return redirect(url_for("index"))

    # Create unique run id for outputs
    run_id = uuid.uuid4().hex[:10]
    out_prefix = os.path.join(app.config["UPLOAD_FOLDER"], run_id)
    os.makedirs(out_prefix, exist_ok=True)

    # Save uploaded files temporarily
    pathA = os.path.join(out_prefix, "imageA." + file1.filename.rsplit(".",1)[-1])
    pathB = os.path.join(out_prefix, "imageB." + file2.filename.rsplit(".",1)[-1])
    file1.save(pathA)
    file2.save(pathB)

    # Read & preprocess (limits width)
    A_color = read_and_limit_image(pathA, max_width=MAX_WIDTH)
    B_color = read_and_limit_image(pathB, max_width=MAX_WIDTH)

    # SSIM (returns score and visual BGR image)
    ssim_score, ssim_vis = compute_ssim_vis(A_color, B_color)
    ssim_out_path = os.path.join(out_prefix, "ssim_diff.png")
    ssim_vis_rgb = ssim_vis[:, :, ::-1]  # BGR -> RGB for Pillow saving
    from PIL import Image
    Image.fromarray(ssim_vis_rgb).save(ssim_out_path)

    # ORB matching (returns score, good_matches_count, visual BGR image)
    orb_score, good_count, matches_vis = orb_match_and_visualize(A_color, B_color)
    matches_out_path = os.path.join(out_prefix, "matches.png")
    Image.fromarray(matches_vis[:, :, ::-1]).save(matches_out_path)

    # Colors
    colorsA = dominant_color_names(A_color, k=3)
    colorsB = dominant_color_names(B_color, k=3)
    common_colors = common_color_names(colorsA, colorsB)

    # Combined score (weights: SSIM 0.6, ORB 0.4)
    combined = combine_scores(ssim_score, orb_score)

    # Build context for template
    context = {
        "run_id": run_id,
        "ssim_score": round(float(ssim_score), 4),
        "orb_score": round(float(orb_score), 4),
        "good_matches": int(good_count),
        "combined_percent": round(float(combined) * 100, 2),
        "colorsA": colorsA,
        "colorsB": colorsB,
        "common_colors": common_colors,
        "ssim_img": f"/{ssim_out_path.replace(os.sep, '/')}",
        "matches_img": f"/{matches_out_path.replace(os.sep, '/')}",
        "imgA": f"/{pathA.replace(os.sep, '/')}",
        "imgB": f"/{pathB.replace(os.sep, '/')}"
    }

    return render_template("result.html", **context)

# serve outputs directory (static files already served under /static)
# no extra routes needed

if __name__ == "__main__":
    app.run(debug=True, port=5000)
