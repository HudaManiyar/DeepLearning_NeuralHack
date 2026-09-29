# Deep Learning Labs

Practical exercises from an MSc Deep Learning course, progressing from a single hidden layer to generative models. Each folder has the notebook and a README explaining the aim, method, results and takeaways.

| Lab | Topic | Dataset | Framework | Headline result |
|---|---|---|---|---|
| [Lab 1](Lab1/) | MLP for XOR in three frameworks | XOR truth table | Keras, PyTorch, TensorFlow | All three learn XOR correctly |
| [Lab 2](Lab2/) | Deep feedforward network; depth and activation experiments | Fashion-MNIST | PyTorch | 89.13% test accuracy |
| [Lab 3](Lab3/) | L1, L2 and Elastic Net regularisation | Breast Cancer Wisconsin | scikit-learn | L1: 99.12% val. accuracy, 40% of weights zeroed |
| [Lab 4.1](Lab4/) | CNN from scratch on 500 images | Flowers Recognition | Keras | Shows severe overfitting (96.8% train vs 29% test) |
| [Lab 4.2](Lab4/) | YOLOv5 vs YOLOv8 + Weighted Box Fusion ensemble | Brain Tumour MRI | Ultralytics | Ensemble recall 0.83, fewest missed tumours |
| [Lab 5](Lab5/) | Next-word text generation, RNN vs LSTM | Novel excerpt | Keras | LSTM gives more coherent text |
| [Lab 6](Lab6/) | Sparse autoencoder | CIFAR-10 | PyTorch | 6 : 1 compression with ~5% active neurons |
| [Lab 7](Lab7/) | Variational autoencoder: generation and interpolation | MNIST | PyTorch | Test loss 95.20; generates new digits |

## Running the notebooks
The notebooks were written for **Google Colab** (Lab 4.2 needs a GPU runtime). Lab 4 downloads data from Kaggle: add `KAGGLE_USERNAME` and `KAGGLE_KEY` under **Colab Secrets** (key icon in the left sidebar) before running.
