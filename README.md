🖼️ Image Comparison Web App

A Computer Vision–powered Flask web application that compares two images using SSIM, ORB Feature Matching, and K-Means Color Clustering to generate a detailed similarity analysis with visual outputs.

🚀 Overview

The Image Comparison Web App uses multiple classical Computer Vision algorithms to evaluate and visualize the similarity between two images.
The application includes a modern UI and produces:

✔ SSIM structural similarity score

✔ ORB keypoint feature matching

✔ Dominant color extraction (K-Means)

✔ Common color comparison

✔ SSIM heatmap visualization

✔ ORB feature match image

🎯 Key Features
🔍 SSIM — Structural Similarity

Measures luminance, contrast & structure

Outputs a similarity score (0–1)

Generates a similarity heatmap

🧠 ORB — Feature Matching

Detects keypoints (corners, edges)

Computes descriptors

Generates a feature-match visualization

Computes a normalized match score

🎨 K-Means Color Clustering

Extracts top dominant colors

Compares theme similarity

Finds common colors

🌐 Full Web UI

Clean Bootstrap interface

Live image preview

Instant results dashboard

🧠 Tech Stack

Frontend: HTML, CSS (Bootstrap), JavaScript
Backend: Python, Flask
Computer Vision: OpenCV, scikit-image, scikit-learn, NumPy

📁 Project Structure
Image-Comparison-WebApp/
│
├── app.py
├── requirements.txt
│
├── utils/
│   ├── __init__.py
│   ├── image_utils.py    # SSIM + ORB + preprocessing
│   └── color_utils.py    # K-Means clustering
│
├── templates/
│   ├── index.html        # Upload page
│   └── result.html       # Results dashboard
│
└── static/
    ├── css/
    ├── js/
    └── outputs/          # Auto-generated results

⚙️ How It Works

User uploads Image A and Image B

Flask backend saves inputs

App performs:

SSIM structural similarity

ORB keypoint detection & matching

K-Means dominant color extraction

App generates visuals:

SSIM heatmap

ORB match image

Results are displayed in a clean dashboard

▶️ Run Locally
1️⃣ Install Requirements
pip install -r requirements.txt

2️⃣ Run the App
python app.py

3️⃣ Open Browser
http://127.0.0.1:5000/

  📸 Sample Outputs

  🖥️ Home Page (Upload Screen)
<img width="1366" height="768" alt="Screenshot (22)" src="https://github.com/user-attachments/assets/27765c3b-6594-4974-b732-9c6c541fa985" />

  🖼️ Images Selected
<img width="1366" height="768" alt="Screenshot (23)" src="https://github.com/user-attachments/assets/7b8d3f48-c0eb-4bbe-8b77-4e2eb0bfd5a9" />

  📊 Comparison Results
<img width="1366" height="768" alt="Screenshot (24)" src="https://github.com/user-attachments/assets/95039ab1-8758-47a8-830f-5850fe94ca87" />

 🔍 SSIM Difference Visualization
<img width="1366" height="768" alt="Screenshot (25)" src="https://github.com/user-attachments/assets/217f1b3f-9061-4e61-8a26-7a574275e3d8" />

 🧠 ORB Feature Matching
<img width="1366" height="768" alt="Screenshot (26)" src="https://github.com/user-attachments/assets/a6551acf-70a7-46cc-9884-65344151bf80" />

🧩 Algorithms Used
🔸 SSIM — Structural Similarity

Analyzes: luminance, contrast, structure.

🔸 ORB — (Oriented FAST + Rotated BRIEF)

Keypoint detection

Descriptor extraction

Feature matching

🔸 K-Means Clustering

Groups color pixels

Extracts dominant colors

Compares color similarity

🚀 Future Enhancements

Add CLIP / ResNet deep-learning similarity

Drag-and-drop image upload

Background removal + object-based matching

Deploy on Render/Heroku

🤝 Contributing

Pull requests are welcome!
Fork → Improve → Submit PR

📬 Contact

Developer: M V Karthikeya
Connect on LinkedIn anytime.

📜 License

This project is licensed under the MIT License.
