# Lab 4 — Convolutional Neural Networks and Object Detection

This lab has two parts.

| Part | Notebook | Topic |
|---|---|---|
| 4.1 | [`Lab4.1.ipynb`](Lab4.1.ipynb) | CNN built from scratch for image classification |
| 4.2 | [`Lab4.2.ipynb`](Lab4.2.ipynb) | YOLOv5 vs YOLOv8 object detection, plus an ensemble |

Both parts were limited to **500 images** by the lab brief. Both download data from Kaggle; add `KAGGLE_USERNAME` and `KAGGLE_KEY` to **Colab Secrets** (key icon in the left sidebar) before running.

---

## Lab 4.1 — CNN from Scratch (Flowers)

### Aim
Build and train a CNN without pretrained weights, and observe how it behaves on a very small dataset.

### Dataset
[Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition), 5 classes (daisy, dandelion, rose, sunflower, tulip). 500 images resized to 64×64, split 400 train / 100 test (stratified). Mildly imbalanced: 62 sunflowers vs 132 dandelions.

### Model
Three conv blocks (Conv2D 32 → 64 → 128, each with BatchNorm + MaxPool), then Dense(128) + Dropout(0.4) and a 5-way softmax. **1,143,493 parameters**, Adam, 20 epochs.

### Results
| Metric | Value |
|---|---|
| Training accuracy | 96.8% |
| Validation accuracy | 29.0% |
| **Test accuracy** | **29.0%** |

### Interpretation
This is **severe overfitting**: training accuracy keeps rising while validation loss increases from epoch 2 onward. The cause is 400 images for a 1.1M-parameter model, so it memorises the training set. Misclassifications cluster between visually similar classes (daisy vs dandelion) and lean toward the more frequent classes.

What would fix it: more data, data augmentation, a smaller model, or transfer learning from a pretrained network.

---

## Lab 4.2 — YOLOv5 vs YOLOv8 + Ensemble (Brain Tumour MRI)

### Aim
Train two object-detection models to locate brain tumours in MRI scans, compare them, and test whether combining them improves results.

### Dataset
[Brain Tumor (Ultralytics)](https://www.kaggle.com/datasets/ultralytics/brain-tumor): MRI images with bounding boxes in YOLO format. 400 train / 100 validation.

### Models
Both fine-tuned from COCO weights for 50 epochs at 640 px, batch size 16.

| | YOLOv5s | YOLOv8n |
|---|---|---|
| Detection | Anchor-based | Anchor-free |
| Head | Coupled | Decoupled (separate class and box) |
| Backbone block | C3 | C2f |

The ensemble uses **Weighted Box Fusion (WBF)**, which averages overlapping boxes from both models weighted by confidence, instead of keeping only the top box as NMS does.

### Results — validation metrics
| Metric | YOLOv5s | YOLOv8n |
|---|---|---|
| mAP@0.5 | **0.4605** | 0.4442 |
| mAP@0.5:0.95 | 0.3181 | **0.3344** |
| Precision | 0.4612 | **0.4772** |
| Recall | **0.8018** | 0.6826 |

### Results — detection counts including the ensemble
| Model | Precision | Recall | F1 | Found (TP) | False alarms (FP) | Missed (FN) |
|---|---|---|---|---|---|---|
| YOLOv5s | 0.6639 | 0.7248 | 0.6930 | 79 | 40 | 30 |
| YOLOv8n | **0.6667** | 0.7890 | **0.7227** | 86 | 43 | 23 |
| WBF Ensemble | 0.6067 | **0.8349** | 0.7027 | **91** | 59 | **18** |

### Interpretation
- Neither model dominates: YOLOv5s has the better mAP@0.5, and YOLOv8n is better at stricter overlap thresholds and on F1.
- The ensemble **found the most tumours and missed the fewest** (18 misses vs 23 for the best single model), but it produced more false alarms, so its **F1 was 2.77% lower** than YOLOv8n's.
- In medical screening, a missed tumour is usually worse than a false alarm, so the ensemble's higher recall can still be the preferred trade-off, but it did not improve overall F1.
