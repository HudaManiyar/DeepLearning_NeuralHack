# Lab 5 — Next-Word Text Generation with RNN and LSTM

**Notebook:** [`Lab5.ipynb`](Lab5.ipynb)

## Aim
Train recurrent networks to predict the next word in a sentence, use them to generate new text, and compare a simple RNN with an LSTM.

## Data
A short text passage from a novel, uploaded as `novel.txt` (vocabulary of **383 words**). The text was lowercased to keep the vocabulary small.

## Preprocessing
1. **Tokenise**: each unique word gets an integer index.
2. **Build n-gram sequences**: every sentence becomes progressively longer prefixes ("the rain", "the rain had", …), each paired with the word that comes next.
3. **Pad** sequences at the front so they all have the same length.
4. Split into input `X` (all words but the last) and target `y` (the next word, one-hot encoded).

## Models
| Model | Architecture |
|---|---|
| RNN | Embedding(100) → SimpleRNN(150) → Dense(softmax over vocabulary) |
| LSTM | Embedding(100) → LSTM(150) → Dense(softmax over vocabulary) |

Both use categorical cross-entropy and Adam, trained for 50 epochs.

## Text generation
Starting from a seed phrase, the model predicts a probability for every word, samples one using **temperature = 0.8** (lower is safer and more repetitive; higher is more random), appends it, and repeats.

Example (seed: *"the rain had"*):
> the rain had settled stopped for three days days surface gray dream intensified among parted themselves behind moved

## Results
Both models reached roughly **80–84% training accuracy** on next-word prediction. The LSTM's output was more coherent, because its gates let it carry context across more words than a simple RNN.

## Takeaways
- An LSTM handles longer-range dependencies better than a SimpleRNN, which suffers from vanishing gradients through time.
- With a small corpus the model mostly recombines phrases it has seen, which explains the grammatical slips.
- Accuracy here is measured on training data only; a held-out set would be needed to measure generalisation.
