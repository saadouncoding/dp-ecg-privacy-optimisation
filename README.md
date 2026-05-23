# Differential Privacy Budget Optimisation for ECG Classification

Bachelor Thesis — German University in Cairo (GUC), 2026  
**Author:** Zeyad Mohamed Elsayed Hamed Saadoun  
**Supervisor:** Dr. Tamer Mostafa  

## Overview

This repository contains all Kaggle notebooks implementing the experimental 
pipeline for the thesis: *"Differential Privacy Techniques for Protecting 
Patient Data"*.

The project trains a ResNet-18 classifier on the PTB-XL 12-lead ECG dataset 
using DP-SGD (Opacus v1.4.0) and evaluates seven strategies for finding the 
optimal privacy budget ε that best balances diagnostic accuracy (AUROC) and 
privacy protection.

**Key result:** Bayesian Optimisation finds ε* ≈ 0.978 with test AUROC = 0.8513 
(privacy cost 8.61%) using only 8 function evaluations. A sharp tipping point 
exists between the 50/50 and 40/60 fitness weight configurations.

## Dataset

[PTB-XL ECG Dataset](https://physionet.org/content/ptb-xl/1.0.1/) — 21,799 
12-lead ECG recordings, 5 diagnostic superclasses, 10-fold cross-validation.

## Notebooks

### Base Pipeline
| Notebook | Description |
|----------|-------------|
| `dp_sgd_training` | Base DP-SGD training pipeline + no-DP baseline |

### Optimisation Methods (Phase 1 — 3 epochs)
| Notebook | Method | ε* | AUROC |
|----------|--------|----|-------|
| `exploratory_bayesian` | Bayesian Optimisation (exploratory) | 0.806 | 0.8271 |
| `dp_sgd_genetic_algorithm` | Genetic Algorithm | 1.110 | 0.8270 |
| `particle_swarm_optimisation` | Particle Swarm Optimisation | 1.204 | 0.8329 |
| `nsga_ii_v3` | NSGA-II v3 (multi-objective) | 0.798* | 0.8269 |
| `epsilon_constraint` | Epsilon-Constraint | 1.000 | 0.8231 |
| `simulated_annealing_exploratory` | Simulated Annealing (exploratory) | 0.867 | 0.8269 |

### Final Phase (Phase 2 — 20 epochs)
| Notebook | Description | ε* | AUROC |
|----------|-------------|----|-------|
| `bayesian_optimisation_20epochs` | BO final run (70/30 weighting) | 0.9782 | 0.8513 |
| `simulated_annealing_20epochs` | SA final run | 1.1079 | 0.8579 |

### Fitness Weight Sensitivity Analysis
| Notebook | Weight Config | ε* | AUROC |
|----------|---------------|----|-------|
| `bo_60_40` | 60/40 | 0.9782 | 0.8513 |
| `bo_50_50` | 50/50 | 0.9782 | 0.8512 |
| `bo_40_60` | 40/60 ← tipping point | 0.5068 | 0.8339 |
| `bo_30_70` | 30/70 | 0.5068 | 0.8340 |

## Setup

All notebooks run on Kaggle (GPU P100, PyTorch 2.x, Opacus 1.4.0).  
The PTB-XL dataset is available directly as a Kaggle dataset.

## Citation

If you use this code, please cite the thesis (forthcoming).
