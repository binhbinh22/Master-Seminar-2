# Full ML Baseline Report
Date: 2026-09-23T04:08:27.039805Z

## Dataset
- samples: 12775
- features: 41529
- labels: ['DevelopmentNotes', 'Expand', 'Parameters', 'Summary', 'Usage']

## Models and Results

### ovr_logreg
- micro_f1: mean=0.0106, std=0.0019
- macro_f1: mean=0.0106, std=0.0019
- jaccard: mean=0.0076, std=0.0011
- hamming: mean=0.7849, std=0.0046

### classifier_chain
- micro_f1: mean=0.0148, std=0.0015
- macro_f1: mean=0.0119, std=0.0014
- jaccard: mean=0.0148, std=0.0015
- hamming: mean=0.3941, std=0.0006

### ovr_sgd
- micro_f1: mean=0.0000, std=0.0000
- macro_f1: mean=0.0000, std=0.0000
- jaccard: mean=0.0000, std=0.0000
- hamming: mean=0.2000, std=0.0000