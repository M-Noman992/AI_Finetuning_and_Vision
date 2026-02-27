# AI Finetuning and Vision

This repository contains two distinct artificial intelligence and computer vision projects: fine-tuning a Large Language Model (GPT-2) on a custom dataset, and performing object detection (cats and license plates) using OpenCV.

## 🛠 Setup & Run Commands

Open your terminal or command prompt. Install the required dependencies first, then run the scripts directly from the same terminal:

```bash
# 1. Install all required dependencies
pip install transformers datasets torch opencv-python

# 2. Run the GPT-2 Fine-Tuning Script
# This will read the custom dataset, train the model, save it to './fine_tuned_gpt2_model', and output test responses.
python gpt2_finetuning.py

# 3. Run the OpenCV Object Detection Script
# This will process the provided images and save the results with bounding boxes in 'output_cats' and 'output_plates' folders.
python opencv_object_detection.py
