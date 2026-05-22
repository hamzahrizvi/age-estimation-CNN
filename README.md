# Hierarchical 7-Class Balanced Training Experiment

This branch contains an experiment that changed the age-estimation pipeline from the previous 9-class fine-age setup to a more practical 7-class age-group setup.

The goal was to make the model more relevant for age-restriction use cases and reduce unnecessary difficulty around very young age groups.

## Motivation

The previous hierarchical EfficientNet model produced the strongest result so far:

| Metric | Result |
|---|---:|
| Fine age exact accuracy | 48.65% |
| Fine age near-group accuracy (+/-1 group) | 88.62% |
| Coarse age accuracy | 85.47% |
| Gender accuracy | 82.75% |

However, the original fine-age groups were not ideal for the intended use case. Very young ages such as `0–2` and `3–5` had very few samples and were not important for age-restriction style classification.

The dataset was also extremely imbalanced:

| Fine age group | Samples |
|---|---:|
| 0–2 | 158 |
| 3–5 | 153 |
| 6–13 | 4,379 |
| 14–18 | 9,067 |
| 19–24 | 31,844 |
| 25–33 | 76,462 |
| 34–48 | 101,294 |
| 49–64 | 38,628 |
| 65+ | 12,186 |

Because of this, the model performed well on the dominant adult groups but poorly on smaller youth groups.

## Change Made

The fine-age classes were remapped into 7 practical groups:

| Class | Age Range |
|---|---|
| 0 | 0–13 |
| 1 | 14–17 |
| 2 | 18–24 |
| 3 | 25–33 |
| 4 | 34–48 |
| 5 | 49–64 |
| 6 | 65+ |

This keeps the important age-restriction boundary:

```text
under 18 vs 18+
```

while removing unnecessary separation between very young child groups.

The hierarchical model still predicts three outputs:

```text
gender_output       -> 2 classes
coarse_age_output   -> 3 classes
fine_age_output     -> 7 classes
```

## Balanced Downsampling Attempt

A balanced training run was tested using capped samples per fine-age group.

The intent was to reduce overfitting toward the dominant adult classes by limiting the number of samples from very large groups.

A small smoke-test cap produced the following distribution:

```text
fine_age_group
0    158
1    153
2    500
3    500
4    500
5    500
6    500
```

This confirmed that the balancing code worked, but the dataset became too small for meaningful full training.

A larger cap was then used for the main experiment.

## Result

The balanced 7-class experiment did not improve performance.

Observed result around epoch 20:

```text
training fine-age accuracy: ~36.97%
validation fine-age accuracy: ~38.94%
best validation fine-age accuracy: ~39.13%
```

The model plateaued below the previous 9-class hierarchical model.

## Interpretation

This experiment suggests that aggressive downsampling removes too much useful adult data.

Although the original dataset is imbalanced, the adult samples still contain important facial variation that helps the backbone learn useful representations. By reducing those classes too much, the model lost generalisation strength.

The result shows that simple downsampling is not enough to reach the target accuracy.

## Lessons Learned

This experiment was useful because it showed:

- merging very young groups is sensible for the use case
- 7-class age grouping is more practical than 9-class grouping
- strict balancing by downsampling can hurt performance
- the model needs better imbalance handling than simply deleting majority-class samples
- near-group accuracy remains an important metric for this project
- the next improvement should focus on smarter sampling or ordinal-aware loss

## Next Steps

The next planned improvements are:

-weighted class

This branch is kept as an experiment because it documents an important modelling decision: class balancing must be handled carefully, and removing too much majority-class data can reduce model performance.
