EPSTE — Embedded Polygon Symbolic Transfer Entropy

**A geometric-symbolic framework for approximating directed information flow in neural time series**

> MSc Dissertation | Artificial Intelligence and Adaptive Systems | University of Sussex | 2026  
> **Award: Distinction (84%)**  
> Supervisor: Lionel Barnett

---

## Overview

Inferring *directed* causal relationships between brain regions from MEG and EEG recordings is hard. Transfer Entropy (TE) provides a principled, model-free measure of directed information flow — but its practical estimation from finite, noisy neural data is notoriously unstable as embedding dimensionality grows.

**EPSTE** introduces a new representational strategy: rather than estimating TE directly from raw signal amplitudes, it first decomposes each time series into a sequence of *geometric symbolic tokens* derived from local triplets of samples. Each token encodes three complementary aspects of local waveform morphology:

Feature
| Triangle area = Magnitude of change over the triplet |
| Apical angle = Directional correlate of change |
| Central amplitude = Ordinal position in signal space |

These tokens form a compact but expressive symbolic state space. A **GRU-based recurrent neural network** with **attention-based multiple-instance learning (MIL)** is then trained to predict surrogate-validated TE values from bags of symbolic windows.

The core claim: *representational geometry precedes learnability*. Structured symbolic representations that encode multiple complementary morphological features allow the network to resolve subtle but causally relevant signal distinctions — without increasing temporal embedding depth or incurring the exponential data requirements of classical TE estimators.

---

## Key Result

EPSTE was evaluated against a standard symbolic TE baseline using **identical architectures, optimisation procedures, and supervision** — isolating the contribution of representational structure alone.

| Metric                            | EPSTE                     | Baseline |
|
| Pair-level absolute error         | **Lower**                 | Higher |
| Learning curve convergence        | **Faster, lower floor**   | Slow plateau |
| Heatmap structural correspondence | **Sharp, differentiated** | Compressed toward mean |
| Wilcoxon signed-rank (paired)     | **p < 10⁻¹⁴**             | — |

The null hypothesis — *"polygon-based symbolic encoding does not increase learnability more effectively than classical amplitude representations"* — was rejected at p < 10⁻¹⁴.

Performance gains were robust across mean, median, and interquartile ranges, indicating the improvement is not driven by outliers.

---

## Architecture

```
Raw MEG time series (X, Y)
        │
        ▼
┌─────────────────────────────┐
│  Geometric Decomposition     │
│  Local triplets → (area,     │
│  angle, amplitude) per step  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Quantile Binning            │
│  3 continuous features →     │
│  1 integer symbol per step   │
│  (global bins from train set)│
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Symbolic TE Label Gen       │
│  Joint-count estimator       │
│  Max over physiological lags │
│  Miller-Madow bias correction│
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  GRU + Attention MIL         │
│  Separate embeddings X, Y    │
│  Window-level GRU encoding   │
│  Attention pooling → scalar  │
│  MSE regression on TE target │
└─────────────┬───────────────┘
              │
              ▼
        Predicted TE_{X→Y}
```

---

## Data

The study uses source-reconstructed MEG data from a psychedelic research study (Sussex Centre for Consciousness Science), parcellated using the AAL90 atlas (90 regions, 16 channels used), sampled at 600 Hz and downsampled to 250 Hz.

**The MEG dataset is not publicly available** and cannot be included in this repository. To run the full pipeline, you will need MEG data in the following format:

```
MATLAB .mat file containing:
  - signal array: (n_channels, n_timepoints, n_trials)
  - subject identity, drug condition, trial index metadata
```

A **synthetic data demo** is provided (see below) that runs the full EPSTE pipeline on generated signals so you can verify the code without the original dataset.

---

## Synthetic Data Demo

To run EPSTE on synthetic data without the MEG dataset:

```python

import numpy as np
from epste_notebook import (
    triangle_feats_arrays,
    build_bins_from_pairs_subsample,
    metric_series_loop_with_bins,
    surrogate_one_dir
)

# Generate two coupled synthetic time series
np.random.seed(42)
n = 600
x = np.cumsum(np.random.randn(n))           # random walk source
y = 0.6 * np.roll(x, 10) + 0.4 * np.random.randn(n)  # lagged, noisy target

pairs = [(x, y)]

# Build global bins from training pairs
bins = build_bins_from_pairs_subsample(pairs, n_bins=5)

# Compute EPSTE
result = metric_series_loop_with_bins(x, y, bins=bins, lags=(5, 10, 15))
print(f"TE_{{X→Y}}: {result['te_xy']:.4f} bits")

# Run surrogate significance test
surr = surrogate_one_dir(x, y, bins=bins, lags=(5, 10, 15), n_surr=100)
print(f"Observed TE: {surr['te_obs']:.4f} | p-value: {surr['p']:.3f}")
```

---

## Installation

```bash
git clone https://github.com/[your-username]/EPSTE.git
cd EPSTE
pip install -r requirements.txt
```

**Requirements:**
```
numpy
pandas
scipy
torch
matplotlib
mne
openpyxl
```

---

## Repository Structure

```
EPSTE/
├── README.md
├── requirements.txt
├── Dissertation_Py_Notebook.ipynb   # Full pipeline notebook
├── epste_core.py                    # Core functions (extracted from notebook)
│   ├── triangle_feats_arrays()      # Geometric decomposition
│   ├── cats_from_arrays()           # Symbolic tokenisation
│   ├── build_bins_from_pairs_subsample()  # Global bin construction
│   ├── metric_series_loop_with_bins()     # End-to-end EPSTE for a pair
│   └── surrogate_one_dir()          # Phase-randomisation significance test
├── epste_model.py                   # GRU + attention MIL architecture
│   ├── EPSTEDataset                 # PyTorch Dataset for bag-of-windows
│   ├── AttentionMIL                 # Attention pooling module
│   └── EPSTE_GRU                   # Full model
├── epste_eval.py                    # Evaluation, metrics, plotting
│   ├── empirical_report()           # Full evaluation pipeline
│   ├── compare_pair_errors()        # Wilcoxon paired significance
│   └── heatmap_from_pairdf()        # Connectivity heatmaps
└── synthetic_demo.py                # Runnable demo without MEG data
```

---

## Methods Summary

### Geometric Decomposition
Each time series is decomposed into overlapping triplets of consecutive samples. For each triplet (y₁, y₂, y₃), three features are computed:
- **Area**: Heron's formula applied to the triangle formed in (time, amplitude) space
- **Angle**: Arctangent of the slope change between adjacent segments
- **Amplitude**: Central sample value y₂ (ordinal position)

### Symbolic Tokenisation
Each feature dimension is discretised using quantile-based global bin edges learned from training data only. The three bin indices are combined via mixed-radix encoding into a single integer symbol per timestep — producing a symbolic sequence suitable for transition counting and TE estimation.

### Transfer Entropy Estimation
Symbolic TE is estimated using a joint-count estimator with optional Miller-Madow bias correction. TE is computed over a set of physiologically motivated lag values and the maximum is retained as the trial-level target.

### Neural Architecture
A GRU processes bags of symbolic windows (separate embeddings for source and target channels). Attention-based MIL pooling aggregates window-level representations into a trial-level scalar TE prediction. Training uses MSE loss with Adam optimiser and early stopping on validation loss.

### Surrogate Testing
Phase-randomisation surrogates (preserving power spectrum, destroying temporal dependencies) provide a null distribution for significance testing of observed TE values.

---

## Theoretical Contribution

The central claim of this work is that **representational geometry functions as an enabling factor for learning information-theoretic dependencies** — not merely a preprocessing convenience.

By encoding multiple complementary morphological features into each symbolic token, EPSTE increases state separability without requiring higher embedding dimensionality. This mitigates the curse of dimensionality that constrains classical TE estimators in finite-sample neuroimaging settings.

An analogy from language: human listeners do not infer linguistic structure from raw acoustic waveforms — they operate on structured symbolic units (phonemes, syllables) that concentrate relational information into interpretable primitives. EPSTE proposes an analogous symbolic scaffolding for neural time series, where geometric tokens function as the primitive units over which directed causal structure is expressed.

---

## Future Directions

- **Learned symbolisation**: Replace fixed quantile bins with a learnable vector quantisation stage, allowing the model to discover a task-adaptive symbolic vocabulary
- **Richer geometric primitives**: Multiscale curvature descriptors, piecewise polynomial segments, or mixed primitive sets
- **Generative motif modelling**: Train a sequence model on symbolic motif streams (analogous to language modelling) to learn a latent "neural grammar" — enabling generative simulation, anomaly detection, and causal representation learning
- **Multivariate conditional TE**: Condition on additional processes to control for indirect pathways and shared drivers
- **Evolutionary hyperparameter search**: NEAT-style or genetic algorithm optimisation over lag sets, bin counts, and architectural choices
- **Cross-domain application**: The EPSTE framework is domain-agnostic — the geometric tokenisation strategy could apply to any noisy, nonstationary time series where classical TE estimation is intractable (financial, ecological, physiological systems)

---

## Citation

If you use or build on this work, please cite:

```
Finnigan, D.A. (2026). Embedded Polygon Symbolic Transfer Entropy (EPSTE): 
A Geometric Token and Deep Learning Approach to Estimating Transfer Entropy 
in Neuroimaging Time Series. MSc Dissertation, University of Sussex.
```

---

## Author

**David Alexander Finnigan**  
MSc Artificial Intelligence and Adaptive Systems (Distinction)  
BSc Neuroscience with Cognitive Science (2:1)
University of Sussex  

[LinkedIn](https://www.linkedin.com/in/david-alexander-finnigan-6a78b9172/) | davidfinnigan91@gmail.com

---

## Acknowledgements

Supervised by **Lionel Barnett**, Sussex Computational Neuroscience Group. MEG data provided by the Sussex Centre for Consciousness Science.

---

*This repository contains the research code accompanying the MSc dissertation. The MEG dataset used in the study is not publicly available due to participant confidentiality. A synthetic data demonstration is provided for reproducibility.*
