# Lab 1 — XOR with a Multi-Layer Perceptron (Keras, PyTorch, TensorFlow)

**Notebook:** [`Lab1.ipynb`](Lab1.ipynb)

## Aim
Solve the XOR problem with a small neural network, and implement the same network in three frameworks to compare how much each one does for you.

## Why XOR
XOR is the classic example of a problem that is **not linearly separable**: no single straight line separates the outputs `0` from `1`. A single-layer perceptron cannot learn it, so it is the simplest problem that needs a hidden layer.

| x1 | x2 | XOR |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

## What was done
The same architecture was built three times: **2 inputs → 4 hidden units → 1 sigmoid output**, trained with binary cross-entropy and Adam (learning rate 0.1).

| Framework | Style | What had to be written by hand |
|---|---|---|
| **Keras** | High-level `Sequential` API | Nothing beyond the layers: `model.fit()` runs the training loop |
| **PyTorch** | `nn.Module` class | The model class and the full training loop (`zero_grad` → forward → `loss.backward()` → `step`) |
| **TensorFlow (low-level)** | Raw tensors | Weight/bias variables, the forward pass (`tanh`, `sigmoid`, `matmul`), and gradients with `tf.GradientTape` |

A decision-boundary plot was drawn for the Keras model to show the non-linear boundary the hidden layer learns.

## Results
All three implementations predict the XOR truth table correctly: `[0, 1, 1, 0]`.

## Takeaways
- A hidden layer with a non-linear activation is what makes XOR learnable.
- Keras is fastest to write; PyTorch and low-level TensorFlow expose every step of training, which makes backpropagation explicit.
- Learning rate mattered: 0.01 converged too slowly on XOR, so 0.1 was used.
