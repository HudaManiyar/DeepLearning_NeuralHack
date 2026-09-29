# Lab 7 — Variational Autoencoder (VAE) on MNIST (PyTorch)

**Notebook:** [`Lab7.ipynb`](Lab7.ipynb)

## Aim
Build a variational autoencoder that can both reconstruct handwritten digits and **generate new ones**, and explore the structure of its latent space.

## How a VAE differs from an autoencoder
A normal autoencoder maps each image to a single point. A VAE maps each image to a **probability distribution** (a mean μ and variance σ²) and samples from it. A KL-divergence term pulls these distributions toward a standard normal N(0, 1), which makes the latent space smooth and continuous, so random points in it decode into realistic images.

## Dataset
**MNIST**: 70,000 greyscale 28×28 handwritten digits (60,000 train / 10,000 test), flattened to 784 values in [0, 1].

## Model
| Part | Architecture |
|---|---|
| Encoder | 784 → 400 (ReLU) → two heads: μ (20) and log σ² (20) |
| Reparameterisation | z = μ + σ · ε, with ε ~ N(0, 1) |
| Decoder | 20 → 400 (ReLU) → 784 (Sigmoid) |

**Loss = reconstruction (binary cross-entropy) + KL divergence.** Adam, lr = 1e-3, 20 epochs, latent dimension 20.

The **reparameterisation trick** moves the randomness into ε, so gradients can flow through μ and σ during backpropagation; sampling z directly would not be differentiable.

## Results
| Metric (final test) | Value |
|---|---|
| Total loss | 95.20 |
| Reconstruction loss | 69.86 |
| KL divergence | 25.34 |

- Total loss fell from 163.91 to 109.56 in the first 5 epochs, then declined smoothly; train and test curves did not diverge, so there is no sign of overfitting.
- Test loss is slightly *below* training loss because training samples noisy z while evaluation uses z = μ.

## Experiments
1. **Reconstruction**: every reconstructed test digit keeps its identity; fine details such as loops and serifs are softened.
2. **Generation**: 16 images decoded from random z ~ N(0, 1) produce recognisable digits, which shows the KL term shaped the latent space as intended.
3. **Interpolation**: moving in a straight line from the latent code of a "1" to a "7" produces a smooth morph between them.
4. **Latent space plot**: the first two latent dimensions show a roughly Gaussian cloud centred at 0, with partial clustering by digit ("1" separates most clearly).

## Takeaways
- The KL term trades a little reconstruction sharpness for a latent space you can sample from.
- VAEs are generative models; plain autoencoders only compress.
