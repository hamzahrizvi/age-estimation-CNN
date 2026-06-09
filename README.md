## Version 0.4 – Age Regression Head

This update introduces a continuous age regression output alongside the existing hierarchical classification architecture.

### Changes

- Added a dedicated age regression head (`age_output`)
- Extended model outputs to:
  - Gender classification
  - Coarse age classification
  - Fine age classification
  - Continuous age prediction
- Added age MAE evaluation metrics
- Added age-to-group evaluation pipeline
- Added prediction export and regression reporting
- Updated evaluation script to compare:
  - Direct class prediction
  - Age-derived class prediction

### Results

| Metric | Score |
|----------|----------|
| Gender Accuracy | 81.92% |
| Coarse Age Accuracy | 82.44% |
| Fine Age Accuracy | 37.96% |
| Fine Age Near Accuracy (±1 Group) | 79.81% |
| Age-Derived Fine Accuracy | 42.71% |
| Age-Derived Near Accuracy (±1 Group) | 87.97% |
| Age MAE | 8.52 Years |

### Key Findings

The direct fine-age classifier underperformed compared to previous versions. However, converting the predicted continuous age back into age groups produced stronger results.

Notable observations:

- Continuous age estimation achieved an MAE of 8.52 years.
- Age-derived classification outperformed the dedicated fine-age classifier.
- Near-group accuracy remained high at 87.97%.
- The model demonstrates that regression-based age estimation may generalise better than hard age-group classification.

### Technical Notes

Current architecture:

```text
EfficientNetB0 Backbone
        ↓
Shared Feature Layer
        ↓
├── Gender Head (2 classes)
├── Coarse Age Head (3 classes)
├── Fine Age Head (7 classes)
└── Age Regression Head
