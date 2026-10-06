# Task 2 — Yelp Polarity Sentiment Classification

Run mode: **full**

## Team-aligned protocol
- Seed: 42
- Working pool from official training split: 200000
- Validation: 5%
- Test examples: 38000
- Slices: short (<50 words), medium (50–200), long (>200), has_negation, has_contrast

## Models
1. TextCNN
2. DPCNN
3. Scratch Transformer Encoder

All embeddings were learned from scratch. No pretrained LM or embedding was used.

## Metrics

| model       |   accuracy |   roc_auc |   pr_auc |      mcc |   brier_score |        ece |   precision_macro |   recall_macro |   f1_macro |   precision_micro |   recall_micro |   f1_micro |   precision_weighted |   recall_weighted |   f1_weighted |   accuracy_ci_low |   accuracy_ci_high |   macro_f1_ci_low |   macro_f1_ci_high |   mcc_ci_low |   mcc_ci_high |   parameter_count |   training_time_seconds |   mean_training_examples_per_sec |   peak_memory_mb |   mean_gradient_norm |   nan_count |   inference_time_seconds | best_checkpoint                                                              |
|:------------|-----------:|----------:|---------:|---------:|--------------:|-----------:|------------------:|---------------:|-----------:|------------------:|---------------:|-----------:|---------------------:|------------------:|--------------:|------------------:|-------------------:|------------------:|-------------------:|-------------:|--------------:|------------------:|------------------------:|---------------------------------:|-----------------:|---------------------:|------------:|-------------------------:|:-----------------------------------------------------------------------------|
| textcnn     |   0.934447 |  0.982783 | 0.983223 | 0.868896 |     0.0494461 | 0.0201545  |          0.934449 |       0.934447 |   0.934447 |          0.934447 |       0.934447 |   0.934447 |             0.934449 |          0.934447 |      0.934447 |          0.931998 |           0.936711 |          0.931842 |           0.93687  |     0.863682 |      0.874161 |           9970082 |                 46.7163 |                          24409.7 |          393.098 |           inf        |           0 |                  2.48184 | checkpoints/textcnn/best_full.pt     |
| dpcnn       |   0.929684 |  0.982029 | 0.982592 | 0.8595   |     0.0513068 | 0.00942667 |          0.929815 |       0.929684 |   0.929679 |          0.929684 |       0.929684 |   0.929684 |             0.929815 |          0.929684 |      0.929679 |          0.927104 |           0.931974 |          0.926812 |           0.932286 |     0.854245 |      0.864519 |          10378178 |                 37.2216 |                          20427.4 |          538.351 |           inf        |           0 |                  2.60394 | checkpoints/dpcnn/best_full.pt       |
| transformer |   0.920105 |  0.976327 | 0.977266 | 0.840334 |     0.0589676 | 0.00694714 |          0.920229 |       0.920105 |   0.920099 |          0.920105 |       0.920105 |   0.920105 |             0.920229 |          0.920105 |      0.920099 |          0.917394 |           0.922686 |          0.917282 |           0.922895 |     0.835144 |      0.845822 |          14841858 |                158.017  |                           6012.5 |         3268.18  |             0.619873 |           0 |                  5.34095 | checkpoints/transformer/best_full.pt |

## McNemar tests
```json
{
  "textcnn_vs_dpcnn": {
    "b": 1191,
    "c": 1010,
    "chi2": 14.720581553839164,
    "p_value": 0.00012467783685843337
  },
  "textcnn_vs_transformer": {
    "b": 1491,
    "c": 946,
    "chi2": 121.43455067706196,
    "p_value": 0.0
  }
}
```

## Slice metrics

| model       | slice        |     n |   macro_f1 |   error_rate |
|:------------|:-------------|------:|-----------:|-------------:|
| textcnn     | short        |  8941 |   0.926574 |    0.0702382 |
| textcnn     | medium       | 21265 |   0.935057 |    0.0649424 |
| textcnn     | long         |  7794 |   0.934328 |    0.0618424 |
| textcnn     | has_negation | 27868 |   0.931584 |    0.0660973 |
| textcnn     | has_contrast | 22451 |   0.927572 |    0.0717563 |
| dpcnn       | short        |  8941 |   0.925167 |    0.0713567 |
| dpcnn       | medium       | 21265 |   0.931565 |    0.0684223 |
| dpcnn       | long         |  7794 |   0.921097 |    0.0742879 |
| dpcnn       | has_negation | 27868 |   0.925505 |    0.0721975 |
| dpcnn       | has_contrast | 22451 |   0.919412 |    0.0799519 |
| transformer | short        |  8941 |   0.916447 |    0.0803042 |
| transformer | medium       | 21265 |   0.921231 |    0.0787679 |
| transformer | long         |  7794 |   0.911528 |    0.0824994 |
| transformer | has_negation | 27868 |   0.912287 |    0.0843979 |
| transformer | has_contrast | 22451 |   0.908061 |    0.0909091 |

Best model by macro-F1: **textcnn**.

## Interpretation
TextCNN achieved the strongest overall performance, with 93.44% accuracy, a macro-F1 of 0.9344, ROC-AUC of 0.9828, and MCC of 0.8689. DPCNN was competitive but slightly weaker, while the scratch Transformer produced the lowest accuracy and macro-F1 and required substantially more training time.

The paired McNemar tests indicate that TextCNN's prediction differences relative to both DPCNN and the Transformer are statistically significant. TextCNN also showed the strongest robustness across most slices. The `has_contrast` slice was one of the more difficult subsets, which agrees with the manual error analysis: reviews containing sentiment reversals such as `but` and `however` frequently caused errors.

The Transformer had the largest parameter count and highest training cost but did not outperform the convolutional models. For this dataset and training setup, the local n-gram features captured by TextCNN were therefore a better accuracy/efficiency trade-off than the scratch Transformer encoder.

Gradient-norm telemetry for TextCNN and DPCNN was recorded as non-finite (`inf`) even though the completed runs reported zero NaN events and produced stable final validation/test metrics. Those gradient-norm values are therefore retained as recorded but should not be interpreted as meaningful finite gradient statistics.