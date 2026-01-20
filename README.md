# Dimensional Analysis in Deep Learning Inference and Physical Constants

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**Author:** MD Yahia Shawon  
**Affiliation:** Institute of High Energy Physics (IHEP), University of Chinese Academy of Sciences (UCAS)

---

## 📄 Abstract

The formulation of interpretable rules from vast computational datasets remains a critical bottleneck in the automation of scientific discovery. This project applies **Dimensional Analysis**—specifically the Buckingham $\pi$ theorem—to address this interpretability gap in both **Artificial Intelligence** and **High Energy Physics**.

By deriving a set of dimensionless groups ($\Pi$) to characterize arithmetic intensity and memory load, this repository demonstrates that complex, high-variance inference latency data collapses onto a single universal curve. Furthermore, this framework is applied to the fine-structure constant ($\alpha$), providing a dimensional argument for its behavior as a dynamic, state-dependent variable.

## 🔑 Key Features

- **Dimensional Scaling Laws:** Derivation of dimensionless groups $\Pi_1$ (Intensity Ratio) and $\Pi_2$ (Dimensionless Load) to predict Transformer performance.
- **Data Collapse:** Empirical proof that normalizing raw latency by $\Pi_2$ reveals a universal, scale-invariant manifold.
- **Physics Isomorphism:** A comparative analysis showing the mathematical equivalence between AI "throughput saturation" and the QED "running" of the fine-structure constant $\alpha$.
- **Profiling Protocol:** A robust PyTorch-based measurement protocol (Algorithm 1) for gathering clean latency data across logarithmic batch/sequence grids.

## 🛠️ Methodology: The $\Pi$ Groups

We treat the Transformer inference process as a physical flow constrained by hardware resources. The system is governed by two primary dimensionless numbers:

### 1. Arithmetic Intensity Ratio ($\Pi_1$)
$$\Pi_1 = \frac{P_{mem} \cdot C}{2N \cdot B \cdot L \cdot M}$$
* **Physical Meaning:** The ratio of memory access time to computation time.
* **Interpretation:** * $\Pi_1 \ll 1$: **Memory-Bound** regime (bandwidth limited).
  * $\Pi_1 \gg 1$: **Compute-Bound** regime (FLOPs limited).

### 2. Dimensionless Load ($\Pi_2$)
$$\Pi_2 = \frac{B \cdot L}{N}$$
* **Physical Meaning:** The relative scale of the input tensor compared to model capacity.
* **Interpretation:** Acts as the state variable for the system's efficiency curve.

---

## 📊 Results

### 1. The "Curse of Dimensionality" vs. Universal Collapse
Raw inference data (left) exhibits high variance and scattering. By applying our dimensional transformation, the data **collapses** onto a single universal curve (right), revealing the fundamental physics of the hardware-software interaction.

| Raw Data (High Variance) | Data Collapse (Universal Law) |
| :---: | :---: |
| <img src="results_figure.png" alt="Raw Data vs Collapse" width="100%"> |

### 2. Mechanism of Saturation ($\Pi_1$)
Analysis of the Arithmetic Intensity ratio ($\Pi_1$) explains *why* the collapse occurs. At low intensity, memory overheads dominate cost. As intensity increases, the system hits the hardware's theoretical floor.

<div align="center">
  <img src="pi1_analysis.png" alt="Arithmetic Intensity Analysis" width="70%">
</div>

### 3. The Physics Connection: Running $\alpha$
We demonstrate that the behavior of the fine-structure constant $\alpha$ under renormalization is mathematically isomorphic to the scaling of AI inference efficiency.

<div align="center">
  <img src="fig1_running_coupling.png" alt="Running Alpha" width="60%">
</div>

---

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yahiashawon/SimpleTransformer-dm-anlys.git](https://github.com/yahiashawon/SimpleTransformer-dm-anlys.git)
   cd SimpleTransformer-dm-anlys
