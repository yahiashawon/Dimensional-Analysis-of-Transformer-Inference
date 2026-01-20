import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set academic plotting style for the report
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.5
})

class PhysicsEngine:
    """
    Handles the theoretical calculation of the Running Coupling Constant.
    This generates the data for the 'Physics' part of your report (Section 3).
    """
    def __init__(self):
        self.alpha_0 = 1.0 / 137.036
        self.m_e = 0.511e-3  # GeV
        self.m_Z = 91.1876   # GeV

    def get_running_coupling(self, Q):
        # QED Beta Function (1-loop approximation)
        # alpha(Q) increases as energy scale Q increases
        Q = np.maximum(Q, self.m_e)
        beta_term = (self.alpha_0 / (3 * np.pi)) * np.log((Q**2) / (self.m_e**2))
        return self.alpha_0 / (1 - beta_term)

def plot_running_coupling_figure():
    """
    Generates Figure 1: The Physics Theory
    Demonstrates that Alpha varies with scale (Professor's requirement).
    """
    print("Generating Figure 1 (Physics Theory)...")
    engine = PhysicsEngine()
    Q_range = np.logspace(-3, 3, 500) # From 1 MeV to 1 TeV
    alpha_eff = engine.get_running_coupling(Q_range)
    inv_alpha = 1.0 / alpha_eff

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Q_range, inv_alpha, color='#d62728', label=r'Inverse Coupling $1/\alpha(Q)$')
    
    # Annotations for the Report
    ax.axvline(x=engine.m_e, color='gray', linestyle='--', alpha=0.5)
    ax.text(engine.m_e * 1.5, 136.5, r'$Q=m_e$ (Rest Mass)', fontsize=10)
    ax.axvline(x=engine.m_Z, color='gray', linestyle='--', alpha=0.5)
    ax.text(engine.m_Z * 0.05, 129.5, r'$Q=m_Z$ (High Energy)', fontsize=10)

    ax.set_xscale('log')
    ax.set_xlabel(r'Energy Scale $Q$ [GeV]', fontsize=12)
    ax.set_ylabel(r'Inverse Fine-Structure Constant $\alpha^{-1}$', fontsize=12)
    ax.set_title(r'Renormalization: The "Running" Coupling $\alpha(Q^2)$', fontsize=14, pad=15)
    ax.grid(True, which="both", alpha=0.15)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('fig1_running_coupling.png', dpi=300)
    print("✓ Saved 'fig1_running_coupling.png'")

def plot_ai_collapse_figure(csv_path):
    """
    Generates Figure 2: The AI Experiment
    Loads YOUR existing CSV and proves the Dimensionless Collapse.
    """
    print(f"Generating Figure 2 (AI Experiment) from {csv_path}...")
    try:
        # LOAD YOUR EXPERIMENTAL DATA
        df = pd.read_csv(csv_path)
        
        # Verify columns exist
        required = ['Pi_2', 'latency_per_token_ms', 'batch_size']
        if not all(col in df.columns for col in required):
            raise ValueError(f"CSV missing columns. Found: {df.columns}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- Subplot 1: The "Messy" Raw Data ---
        # Plotting Raw Latency vs Batch Size (Standard Benchmark View)
        # Note: We group by sequence length to show the "spread"
        seq_lens = df['seq_len'].unique()
        for sl in seq_lens:
            subset = df[df['seq_len'] == sl]
            ax1.plot(subset['batch_size'], subset['latency_ms'], 'o-', 
                     label=f'Seq {sl}', alpha=0.7)
            
        ax1.set_xscale('log', base=2)
        ax1.set_yscale('log')
        ax1.set_xlabel('Batch Size (B)', fontsize=12)
        ax1.set_ylabel('Total Latency [ms]', fontsize=12)
        ax1.set_title(r'(a) Raw Data: The Curse of Dimensionality', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8, loc='upper left')

        # --- Subplot 2: The "Clean" Dimensionless Collapse ---
        # Plotting YOUR calculated Pi_2 vs Normalized Latency
        # This proves the "Physical Law" of the AI
        sc = ax2.scatter(df['Pi_2'], df['latency_per_token_ms'], 
                         c=df['batch_size'], cmap='plasma', 
                         s=100, edgecolor='k', alpha=0.8)
        
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel(r'Dimensionless Load $\Pi_2 = \frac{B \times L}{N}$', fontsize=12)
        ax2.set_ylabel(r'Latency per Token [ms]', fontsize=12)
        ax2.set_title(r'(b) Data Collapse: The Universal Law', fontsize=13)
        ax2.grid(True, which="major", ls="--", alpha=0.3)
        
        cbar = plt.colorbar(sc, ax=ax2)
        cbar.set_label('Batch Size', fontsize=10)

        plt.tight_layout()
        plt.savefig('fig2_dimensionless_collapse.png', dpi=300)
        print("✓ Saved 'fig2_dimensionless_collapse.png'")

    except Exception as e:
        print(f"Error reading CSV: {e}")
        print("Please ensure 'llm_inference_results.csv' is in the same folder.")

if __name__ == "__main__":
    # 1. Plot the Physics Theory (Math-based)
    plot_running_coupling_figure()
    
    # 2. Plot the AI Experiment (Data-based)
    plot_ai_collapse_figure('llm_inference_results.csv')