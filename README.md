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
