# Lab 3 — L1, L2 and Elastic Net Regularisation

**Notebook:** [`Lab3.ipynb`](Lab3.ipynb)

## Aim
Compare how L1, L2 and Elastic Net regularisation change a model's weights, loss and generalisation, and how the regularisation strength λ controls that trade-off.

## Dataset
**Breast Cancer Wisconsin** (scikit-learn): 569 samples, 30 numeric features, binary label (malignant / benign). Features were standardised first, because L1/L2 penalties depend on weight size and unscaled features would be penalised unfairly.

## Model
Logistic regression, chosen because its weights, loss and cost are easy to inspect directly. In scikit-learn, regularisation strength is set through `C`, where **λ = 1 / C**.

## Results
| Model | Train acc. | Val. acc. | Train loss | Val. loss | Zero weights (of 30) |
|---|---|---|---|---|---|
| Baseline (no penalty) | 98.90% | 96.49% | 0.0286 | 0.1042 | 0 |
| **L1** | 98.90% | **99.12%** | 0.0537 | **0.0752** | **12 (40%)** |
| L2 | 98.90% | 98.25% | 0.0513 | 0.0777 | 0 |
| Elastic Net (50/50) | 98.90% | 98.25% | 0.0526 | 0.0767 | 3 |

- All three penalties **lowered validation loss** compared with the baseline, i.e. they reduced overfitting.
- **L1 performed feature selection**, driving 12 of 30 weights to exactly zero while giving the best validation accuracy.
- **L2 shrank weights smoothly** without zeroing any.
- **Elastic Net** sat in between: some sparsity (3 zeros) plus L2's stability.

## Sensitivity to λ
Each method was retrained for `C ∈ {0.01, 0.1, 1, 10, 100}` and the training/validation loss, number of zero weights and weight norm were plotted.
- Small C (strong regularisation): more zero weights and smaller norms, but higher loss (underfitting).
- Large C (weak regularisation): the model approaches the unregularised baseline.

## Takeaways
- Use **L1** when you want a sparse, interpretable model; **L2** when all features are likely useful; **Elastic Net** when features are correlated and you want a mix.
- Regularisation strength is a hyperparameter: too much underfits, too little overfits.
