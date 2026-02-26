# Tree-Based Learning Algorithms from Scratch

## Project Overview

This project implements:

1. Decision Tree Classifier (from scratch)
2. Random Forest Classifier (built using custom Decision Trees)
3. Experimental comparison with sklearn implementations

The objective is to demonstrate deep algorithmic understanding,
mathematical correctness, and clean software engineering practices.

---

# Mathematical Background

## Gini Impurity

Gini impurity measures node impurity:

G = 1 - Σ p_i²

where p_i is the probability of class i in a node.

## Information Gain

Information Gain = G(parent) - Weighted G(children)

The split maximizing Information Gain is selected.

---

## Bagging (Bootstrap Aggregation)

Random Forest uses bootstrap sampling:

- Sample with replacement
- Train multiple trees independently
- Aggregate predictions via majority voting

This reduces variance while maintaining low bias.

---

## Bias-Variance Discussion

- Decision Trees → Low bias, high variance
- Random Forest → Reduced variance through averaging
- Tradeoff managed via max_depth, n_estimators

---

# Algorithm Design

### Decision Tree
- Recursive binary splits
- Exhaustive search over features
- Gini-based split selection
- Stopping conditions:
  - max_depth
  - min_samples_split
  - pure node

### Random Forest
- Bootstrap sampling
- Random feature subset at each split
- Majority voting

---

# Experimental Results

Dataset: Breast Cancer (sklearn)

Comparison includes:
- Train Accuracy
- Test Accuracy
- Training Time

Plots generated using matplotlib.

---

# Time Complexity

Decision Tree:
O(n_features × n_samples²)

Random Forest:
O(n_estimators × n_features × n_samples²)

---

# Comparison with sklearn

- Custom implementations closely match sklearn accuracy
- Sklearn is faster due to C optimizations
- Algorithmic behavior remains consistent

---

# Future Improvements

- Add entropy criterion
- Implement pruning
- Add multi-class support
- Parallelize forest training
- Add feature importance calculation

---

# How to Run

```bash
pip install -r requirements.txt
python experiment.py
