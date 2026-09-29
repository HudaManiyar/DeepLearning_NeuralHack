# Lab 6 — Sparse Autoencoder on CIFAR-10 (PyTorch)

**Notebook:** [`Lab6.ipynb`](Lab6.ipynb)

## Aim
Build a sparse autoencoder that learns a compressed representation of images in which only a few hidden neurons are active for any one input, and study how the sparsity penalty affects reconstruction.

## Concept
A plain autoencoder can learn to simply copy its input. A **sparse** autoencoder adds a penalty that keeps most hidden neurons near zero, so each image is represented by a small set of selective features. This mirrors sparse coding in the visual cortex, where only a small fraction of neurons fire at once.

**Loss = reconstruction MSE + β × KL divergence**, where the KL term pushes each neuron's average activation toward a small target ρ = 0.05 (about 5% of neurons active).

## Dataset
**CIFAR-10**: 60,000 colour 32×32 images in 10 classes (50,000 train / 10,000 test). Each image is flattened to 3,072 values and normalised per channel.

## Model
| Layer | Size | Activation |
|---|---|---|
| Encoder 1 | 3072 → 1024 | ReLU |
| Encoder 2 (code) | 1024 → 512 | ReLU |
| Decoder 1 | 512 → 1024 | ReLU |
| Decoder 2 | 1024 → 3072 | Sigmoid |

That is a **6 : 1 compression** (3072 → 512). Trained with Adam (lr = 1e-3), batch size 128, 30 epochs, β = 1e-3.

## What was done
1. Trained the autoencoder and plotted total, reconstruction and sparsity loss per epoch.
2. Evaluated reconstruction loss on the test set.
3. Compared original and reconstructed test images.
4. Plotted the 512-dimensional hidden activations for several images to check that most neurons are inactive and that different classes activate different neurons.
5. **Sparsity experiment**: retrained for 10 epochs with β = 0, 1e-4, 1e-3 and 1e-2 to compare reconstruction quality against sparsity.

## Expected behaviour
| β | Reconstruction | Sparsity |
|---|---|---|
| 0 | Sharpest | None; all neurons fire |
| 1e-4 | Slightly softer | Mild |
| **1e-3** | Balanced | Close to the 5% target |
| 1e-2 | Blurriest | Very high |

**Note:** the saved notebook does not include its run outputs, so the numeric results are not reproduced here. Re-run the notebook to regenerate the loss curves, test MSE and reconstruction images.

## Takeaways
- Sparsity trades reconstruction quality for a more selective, interpretable representation.
- β around 1e-3 balances the two for this setup.
- A fully connected autoencoder blurs fine detail at 6 : 1 compression; a convolutional autoencoder would preserve more spatial structure.
