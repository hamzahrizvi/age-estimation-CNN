## Weighted-Loss Experiment

A weighted-loss version of the hierarchical EfficientNet model was tested to address class imbalance in the fine-age groups.

The goal was to improve recall for underrepresented age groups without downsampling the full dataset. Instead of removing adult samples, the training loop applied higher sample weights to minority fine-age classes.

### Motivation

The best v2 model showed strong near-group behaviour but weaker exact fine-age classification:

| Metric | Result |
|---|---:|
| Fine age exact accuracy | 48.65% |
| Fine age near-group accuracy (+/-1 group) | 88.62% |
| Coarse age accuracy | 85.47% |
| Gender accuracy | 82.75% |

The dataset was heavily imbalanced, especially toward adult age groups:

| Fine age group | Samples |
|---|---:|
| 0–13 | 311 |
| 14–17 | 9,067 |
| 18–24 | 31,844 |
| 25–33 | 76,462 |
| 34–48 | 101,294 |
| 49–64 | 38,628 |
| 65+ | 12,186 |

### Result

The weighted-loss run did not improve performance. The best validation fine-age accuracy was approximately:

```text
28.30%
```

The run stopped early after validation accuracy failed to improve:

```text
Final validation fine-age accuracy: 28.10%
Best validation fine-age accuracy: 28.30%
```

### Interpretation

This experiment likely failed because the minority classes were extremely small compared with the adult classes. The weighting made the model over-emphasise noisy or sparse minority groups, which damaged the shared representation learned by EfficientNet.

The result suggests that simple class weighting is not enough for this dataset. Better approaches would be:

- targeted data collection for underrepresented age groups
- softer ordinal age labels
- near-group-aware loss functions
- controlled oversampling instead of aggressive weighting
- separate binary age-restriction classification, such as under-18 vs 18+
- staged training where the model first learns broad age bands, then fine age groups

This branch is kept as an experiment to show the evaluation process and why weighted loss was not selected as the best model.
````
