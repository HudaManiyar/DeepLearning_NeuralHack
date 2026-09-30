# Lab 2 — Deep Feedforward Network on Fashion-MNIST (PyTorch)

**Notebook:** [`Lab2.ipynb`](Lab2.ipynb)

## Aim
Design, train and evaluate a deep feedforward neural network for image classification, and study how **network depth** and **activation functions** affect learning.

## Dataset
**Fashion-MNIST**: 70,000 greyscale 28×28 images of clothing in 10 classes (60,000 train / 10,000 test).

## Model
Fully connected network: **784 → 128 → 64 → 32 → 10**, ReLU activations, cross-entropy loss, Adam (lr = 0.001), 50 epochs.

## Results
| Metric | Value |
|---|---|
| Training accuracy (epoch 50) | 95.76% |
| **Test accuracy** | **89.13%** |
| Training loss | 594.11 → 104.41 (epoch 1 → 50) |

The ~6.6-point gap between training and test accuracy shows **overfitting**: the network began memorising training images rather than learning only general patterns.

A plot of first-hidden-layer activations for one image shows many neurons at exactly zero, which is ReLU clipping negative values.

## Experiments
**1. Network depth** (3 epochs each)

| Hidden layers | Accuracy after 3 epochs |
|---|---|
| 1 | 87.05% |
| 3 | 86.97% |
| 5 | 86.17% |

Deeper networks started slower and had not caught up within 3 epochs; for a dataset this simple, extra depth adds optimisation difficulty without an early benefit.

**2. Activation functions**: ReLU, Sigmoid, Tanh and Leaky ReLU were compared on the same network.
- ReLU and Leaky ReLU converged fastest and reached the highest accuracy.
- Sigmoid and Tanh trained more slowly because of saturation and vanishing gradients.
- Leaky ReLU avoids "dead" neurons by letting a small gradient through for negative inputs.

## Takeaways
- A simple MLP reaches ~89% on Fashion-MNIST, but overfits without regularisation.
- More layers are not automatically better; they need more training to pay off.
- ReLU-family activations are the default for a reason: they keep gradients alive.
