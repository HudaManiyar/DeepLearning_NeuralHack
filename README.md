# Deep Learning 
### MSc. Artificial Intelligence & Machine Learning

---

## Final Exam — NeuralHack 2026

### Smart Garbage Classification System

A deep learning system that classifies garbage images into 6 waste categories and determines whether the item is recyclable or non-recyclable. Built as part of the NeuralHack 2026 hackathon examination for the Deep Learning course.


<img width="1163" height="517" alt="image" src="https://github.com/user-attachments/assets/eaef9e1a-b839-4037-940c-16f3247bc106" />

---

## Problem Statement

Improper waste disposal is a critical environmental challenge. Manual sorting of garbage is slow, inconsistent, and unsustainable at scale. This system automates waste classification from images, enabling smart bin deployment and automated recycling pipelines.

- **Input:** RGB image of a waste item (224x224x3)
- **Output:** One of 6 waste categories — Cardboard, Glass, Metal, Paper, Plastic, Trash
- **Task Type:** Multi-class Image Classification (Supervised Learning)
- **Real-world Impact:** Supports SDG Goal 12 — Responsible Consumption and Production

---

## Model Architecture

A Custom Convolutional Neural Network (CNN) trained from scratch using TensorFlow and Keras.

| Layer | Type | Filters / Units | Activation | Regularization |
|---|---|---|---|---|
| Block 1 | Conv2D + BatchNorm + MaxPool | 32 | ReLU | L2 = 0.001 |
| Block 2 | Conv2D + BatchNorm + MaxPool | 64 | ReLU | L2 = 0.001 |
| Block 3 | Conv2D + BatchNorm + MaxPool | 128 | ReLU | L2 = 0.001 |
| Block 4 | Conv2D + BatchNorm + MaxPool | 256 | ReLU | L2 = 0.001 |
| Head | GlobalAveragePooling2D | — | — | — |
| Dense 1 | Dense + Dropout | 512, p=0.5 | ReLU | Dropout |
| Dense 2 | Dense + Dropout | 256, p=0.3 | ReLU | Dropout |
| Output | Dense | 6 | Softmax | — |

**Total Parameters:** 654,790  
**Loss Function:** Categorical Cross-Entropy  
**Optimizer:** Adam (lr = 0.001)

---

## Results

| Metric | Value |
|---|---|
| Training Accuracy | 81.62% |
| Validation Accuracy | 70.58% |
| Generalization Gap | 11.04% |
| Sample Test (12 images) | 10/12 correct (83.3%) |

---

## Project Structure

```
Final_Exam/
├── DL_ModelTraining.ipynb     # Complete training notebook (Google Colab)
├── app.py                     # Streamlit UI application
├── garbage_classifier.h5      # Trained model weights
├── class_names.json           # Class label mapping
└── training_history.json      # Accuracy and loss data per epoch
```

---

## Dataset

**Garbage Classification Dataset** — Kaggle  
- 2527 labeled images across 6 waste categories  
- Source: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification  
- Split: 80% training (2024 images) / 20% validation (503 images)

---

## How to Run the Streamlit App

**1. Clone the repository**
```bash
git clone https://github.com/HudaManiyar/Deep-Learning.git
cd Deep-Learning/Final_Exam
```

**2. Create and activate a virtual environment**
```bash
python -m venv dl_env
dl_env\Scripts\activate        # Windows
source dl_env/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install streamlit tensorflow pillow plotly pandas
```

**4. Run the app**
```bash
streamlit run app.py
```

**5. Open in browser**
```
http://localhost:8501
```

---

## UI Features

- Upload any image of a waste item
- Displays predicted waste category with confidence percentage
- Shows whether the item is **Recyclable** or **Non-Recyclable**
- Displays a confidence bar chart across all 6 classes
- Shows a disposal tip for the predicted category
- Includes CNN architecture table and interactive training history plots

---

## Future Improvements

- Transfer Learning using MobileNetV2 — expected to improve accuracy to 85-92%
- K-Fold Cross Validation to reduce validation instability
- Collect more Trash and Plastic images to address class imbalance

---

## Labs

The `LABS/` folder contains all practical exercises completed during the Deep Learning course covering:

- Multi-Layer Perceptron (MLP) in Keras
- Deep Feedforward Networks
- L1 and L2 Regularization
- Convolutional Neural Networks
- LSTM (Long Short-Term Memory)
- Sparse and Contractive Autoencoders

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![Keras](https://img.shields.io/badge/Keras-3.13-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-brightgreen)
![Plotly](https://img.shields.io/badge/Plotly-5.x-blue)
![Colab](https://img.shields.io/badge/Google_Colab-T4_GPU-yellow)
