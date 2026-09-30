# Classical ML Baseline (corrected multi-label data)
Date: 2026-09-24T14:49:19.363928Z

## Dataset
- samples (unique comments): 2555
- labels: ['DevelopmentNotes', 'Expand', 'Parameters', 'Summary', 'Usage']
- folds: 5

## Results

### ovr_logreg
- micro_f1: mean=0.6282, std=0.0116
- macro_f1: mean=0.5928, std=0.0090
- jaccard: mean=0.5843, std=0.0137
- hamming: mean=0.1697, std=0.0052
- precision_micro: mean=0.6172, std=0.0103
- recall_micro: mean=0.6397, std=0.0140

### classifier_chain_logreg
- micro_f1: mean=0.6098, std=0.0122
- macro_f1: mean=0.5667, std=0.0118
- jaccard: mean=0.5977, std=0.0144
- hamming: mean=0.1748, std=0.0058
- precision_micro: mean=0.6103, std=0.0108
- recall_micro: mean=0.6093, std=0.0143

### ovr_linear_svc
- micro_f1: mean=0.6236, std=0.0108
- macro_f1: mean=0.5854, std=0.0092
- jaccard: mean=0.5801, std=0.0121
- hamming: mean=0.1709, std=0.0050
- precision_micro: mean=0.6160, std=0.0107
- recall_micro: mean=0.6313, std=0.0120

### ovr_sgd
- micro_f1: mean=0.6266, std=0.0121
- macro_f1: mean=0.5932, std=0.0097
- jaccard: mean=0.5876, std=0.0149
- hamming: mean=0.1746, std=0.0060
- precision_micro: mean=0.6018, std=0.0127
- recall_micro: mean=0.6536, std=0.0149