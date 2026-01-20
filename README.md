# Dimensional Analysis of Transformer Inference

This project applies classical dimensional analysis to a transformer-based language model inference workload to identify compute–memory regimes using dimensionless groups. The code measures latency and throughput for different batch sizes and sequence lengths, computes dimensionless groups, and produces figures suitable for inclusion in a paper or presentation. [file:91][file:93]

---

## 1. Project Overview

Large language model (LLM) inference performance depends on model size, batch configuration, sequence length, and hardware characteristics. This repository implements a small transformer benchmark and uses dimensional analysis to construct dimensionless groups that describe the system’s behavior, such as:

- Arithmetic intensity ratio \(\Pi_1 = t_c / t_m\) (compute time vs memory time).  
- Activation–parameter ratio \(\Pi_2 = BL/N\).  
- Batching efficiency \(\Pi_3\) (empirical). [file:91]

The goal is to show that plotting performance against these groups reveals clear regimes (for example, compute‑bound) and allows data collapse across different configurations. [file:91][file:93]

---

## 2. Repository Structure

```text
project-root/
├─ llm_dimensional_analysis.py   # Main benchmark + plotting script
├─ llm_inference_results.csv     # Generated measurements
├─ results_figure.png            # Main 3-panel figure (latency, throughput, collapse)
├─ pi1_analysis.png              # Additional arithmetic intensity figure
├─ paper/                        # (Optional) LaTeX/Word files for the report
└─ README.md                     # This file
