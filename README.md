# GRB_Technology — Virtual Clothing Try-On

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PM2003/GRB_Technology/blob/main/setup_colab.ipynb)

An AI-powered virtual clothing try-on system built on the ViTON architecture. Upload a person photo and a clothing item — the system generates a photorealistic result of the person wearing that clothing.

## How It Works

```
Person Photo + Clothing Image
        ↓
  Preprocessing Pipeline
  ├── Body Parsing (SCHP)
  ├── Pose Estimation (OpenPose)
  └── Cloth Segmentation (U2Net)
        ↓
   ViTON GAN Model
   ├── GMM: Geometric Matching Module (warps cloth to body shape)
   └── TOM: Try-On Module (blends cloth onto person)
        ↓
  Try-On Result Image
```

## Project Structure

```
GRB_Technology/
├── networks/
│   ├── gmm.py                 # Geometric Matching Module
│   ├── tom.py                 # Try-On Module
│   └── u2net.py               # U2Net segmentation network
├── server/
│   ├── api_server.py          # FastAPI inference server
│   ├── preprocessing.py       # Image preprocessing pipeline
│   ├── cloth_segmentation.py  # Cloth masking via U2Net
│   └── viton_inference.py     # End-to-end ViTON inference
├── client/
│   ├── app.py                 # Flask web application
│   ├── templates/index.html   # Web UI
│   └── static/                # CSS + JS
├── utils/
│   ├── image_utils.py
│   └── data_utils.py
├── setup_colab.ipynb          # Google Colab notebook
└── requirements.txt
```

## Quick Start (Google Colab — Recommended)

1. Click **Open in Colab** above
2. Set runtime to **GPU (T4)**
3. Run all cells — a public Gradio URL is generated at the end

## Local Setup

```bash
git clone https://github.com/PM2003/GRB_Technology.git
cd GRB_Technology
pip install -r requirements.txt

# Terminal 1 — start inference server
python -m server.api_server

# Terminal 2 — start web client
cd client && python app.py
```

Open `http://localhost:5000` in your browser.

## Tech Stack

- **ViTON** — Virtual Try-On Network (GAN-based deep learning)
- **U2Net** — Salient object detection for cloth segmentation
- **OpenPose** — Human body keypoint detection
- **SCHP** — Self-Correction Human Parsing
- **FastAPI** — High-performance inference REST server
- **Flask** — Lightweight web client
- **Gradio** — Colab web interface
