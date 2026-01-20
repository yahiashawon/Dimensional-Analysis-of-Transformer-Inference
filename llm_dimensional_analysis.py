"""
Dimensional Analysis of LLM Inference Performance
Measures compute/memory regimes using dimensionless groups
"""

import torch
import torch.nn as nn
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("="*70)
print("  DIMENSIONAL ANALYSIS: LLM INFERENCE PERFORMANCE BENCHMARK")
print("="*70)

# ============================================================================
# 1. SIMPLE TRANSFORMER MODEL
# ============================================================================

class SimpleTransformer(nn.Module):
    """Minimal transformer for performance measurement"""
    def __init__(self, d_model=512, n_heads=8, n_layers=6, vocab_size=50000):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 2048, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embed(x) + self.pos_embed[:, :seq_len, :]
        x = self.transformer(x)
        return self.out(x)


# ============================================================================
# 2. PERFORMANCE MEASUREMENT FUNCTION
# ============================================================================

def measure_latency(model, batch_size, seq_len, device, n_warmup=5, n_runs=20):
    """
    Measure average inference latency with proper warmup
    
    Returns: latency in seconds
    """
    model.eval()
    
    # Generate dummy input
    x = torch.randint(0, 50000, (batch_size, seq_len), device=device)
    
    # Warmup phase (important for GPU)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    
    # Synchronize before measurement
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual measurement
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    
    avg_latency = (end - start) / n_runs
    return avg_latency


# ============================================================================
# 3. HARDWARE SPECIFICATION
# ============================================================================

def get_hardware_specs(device):
    """Get or estimate hardware compute and memory bandwidth"""
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nGPU Detected: {gpu_name}")
        
        # Common GPU specs (TFLOPS for FP32, Memory Bandwidth in GB/s)
        gpu_specs = {
            'RTX 4090': (82.6, 1008),
            'RTX 4080': (48.7, 717),
            'RTX 4070': (29.1, 504),
            'RTX 3090': (35.6, 936),
            'RTX 3080': (29.8, 760),
            'RTX 3070': (20.3, 448),
            'RTX 3060': (12.7, 360),
            'A100': (19.5, 1555),  # FP32, actual tensor core is higher
            'V100': (15.7, 900),
            'T4': (8.1, 320),
        }
        
        # Try to match GPU
        for key, (compute, bandwidth) in gpu_specs.items():
            if key in gpu_name:
                print(f"Matched specs: {compute} TFLOPS, {bandwidth} GB/s")
                return compute, bandwidth
        
        # Default estimate for unknown GPU
        print("Unknown GPU, using conservative estimate")
        return 10.0, 300.0
    else:
        print("\nCPU Detected (performance will be slower)")
        return 1.0, 50.0  # Rough CPU estimate


# ============================================================================
# 4. MAIN BENCHMARK
# ============================================================================

def run_benchmark():
    # Setup device
    device = torch.device('cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Get hardware specs
    C_tflops, M_gbs = get_hardware_specs(device)
    
    # Create model
    print("\nCreating model...")
    model = SimpleTransformer(
        d_model=512,
        n_heads=8,
        n_layers=6,
        vocab_size=50000
    ).to(device)
    
    # Count parameters
    N = sum(p.numel() for p in model.parameters())
    P_gb = (N * 4) / (1024**3)  # Float32 = 4 bytes
    
    print(f"Model Parameters (N): {N:,}")
    print(f"Parameter Memory (P): {P_gb:.3f} GB")
    print(f"Compute Throughput (C): {C_tflops:.1f} TFLOPS")
    print(f"Memory Bandwidth (M): {M_gbs:.1f} GB/s")
    
    # Benchmark configurations
    configs = [
        # (batch_size, sequence_length)
        (1, 32), (1, 64), (1, 128), (1, 256), (1, 512),
        (2, 64), (2, 128), (2, 256), (2, 512),
        (4, 64), (4, 128), (4, 256), (4, 512),
        (8, 64), (8, 128), (8, 256),
        (16, 64), (16, 128), (16, 256),
        (32, 64), (32, 128),
    ]
    
    print(f"\nRunning {len(configs)} benchmark configurations...")
    print("-"*70)
    print(f"{'Batch':>5} {'SeqLen':>6} {'Latency(ms)':>12} {'Tokens/s':>10} {'π₁':>8} {'π₂':>10}")
    print("-"*70)
    
    results = []
    
    for B, L in configs:
        try:
            # Measure latency
            latency = measure_latency(model, B, L, device, n_warmup=5, n_runs=15)
            
            # Compute metrics
            total_tokens = B * L
            throughput = total_tokens / latency
            latency_per_token = (latency * 1000) / total_tokens
            
            # Compute dimensionless groups
            # π₁ = (compute_time) / (memory_time)
            #    = (2*N*B*L / C) / (P / M)
            #    = (2*N*B*L*M) / (C*P)
            compute_ops = 2 * N * B * L  # Approximate FLOPs
            pi_1 = (compute_ops * M_gbs) / (C_tflops * 1e12 * P_gb * 1e9) * 1e12
            
            # π₂ = (activation_size) / (parameter_size) ≈ (B*L) / N
            pi_2 = (B * L) / N
            
            # Store results
            results.append({
                'batch_size': B,
                'seq_len': L,
                'latency_sec': latency,
                'latency_ms': latency * 1000,
                'throughput_tokens_per_sec': throughput,
                'latency_per_token_ms': latency_per_token,
                'total_tokens': total_tokens,
                'Pi_1': pi_1,
                'Pi_2': pi_2,
            })
            
            print(f"{B:5d} {L:6d} {latency*1000:12.2f} {throughput:10.1f} {pi_1:8.3f} {pi_2:10.2e}")
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"{B:5d} {L:6d}  ** OUT OF MEMORY **")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                print(f"{B:5d} {L:6d}  ** ERROR: {str(e)[:30]} **")
            continue
    
    print("-"*70)
    print(f"Completed: {len(results)} / {len(configs)} configurations\n")
    
    return results, N, C_tflops, M_gbs, P_gb


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def create_visualizations(df, N, C_tflops, M_gbs, P_gb):
    """Generate publication-quality figures"""
    
    fig = plt.figure(figsize=(16, 5))
    
    # ---- Panel (a): Raw Latency vs Batch Size ----
    ax1 = plt.subplot(1, 3, 1)
    
    seq_lengths = sorted(df['seq_len'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(seq_lengths)))
    
    for i, L in enumerate(seq_lengths):
        subset = df[df['seq_len'] == L].sort_values('batch_size')
        ax1.plot(subset['batch_size'], subset['latency_ms'], 
                marker='o', linewidth=2.5, markersize=7,
                color=colors[i], label=f'L={L}')
    
    ax1.set_xlabel('Batch Size (B)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Total Latency (ms)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Raw Performance', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log', base=2)
    
    # ---- Panel (b): Throughput ----
    ax2 = plt.subplot(1, 3, 2)
    
    for i, L in enumerate(seq_lengths):
        subset = df[df['seq_len'] == L].sort_values('batch_size')
        ax2.plot(subset['batch_size'], subset['throughput_tokens_per_sec'], 
                marker='s', linewidth=2.5, markersize=7,
                color=colors[i], label=f'L={L}')
    
    ax2.set_xlabel('Batch Size (B)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Throughput (tokens/sec)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Throughput Scaling', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, ncol=2, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log', base=2)
    
    # ---- Panel (c): Dimensionless Representation ----
    ax3 = plt.subplot(1, 3, 3)
    
    scatter = ax3.scatter(
        df['Pi_2'], 
        df['latency_per_token_ms'],
        c=df['batch_size'], 
        cmap='plasma',
        s=150, 
        alpha=0.8, 
        edgecolors='black', 
        linewidth=1
    )
    
    ax3.set_xlabel('$\\Pi_2 = BL/N$ (dimensionless)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Latency per Token (ms)', fontsize=13, fontweight='bold')
    ax3.set_title('(c) Dimensionless Collapse', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, which='both', linestyle='--')
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Batch Size (B)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results_figure.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Figure saved: results_figure.png")
    
    # Additional figure: Pi_1 analysis (if meaningful range)
    if df['Pi_1'].max() / df['Pi_1'].min() > 2:
        fig2, ax = plt.subplots(figsize=(8, 6))
        scatter2 = ax.scatter(df['Pi_1'], df['latency_per_token_ms'],
                            c=df['seq_len'], cmap='coolwarm',
                            s=100, alpha=0.7, edgecolors='black')
        ax.set_xlabel('$\\Pi_1$ = Compute Time / Memory Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latency per Token (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Arithmetic Intensity Analysis', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        cbar2 = plt.colorbar(scatter2, ax=ax)
        cbar2.set_label('Sequence Length', fontsize=11)
        plt.tight_layout()
        plt.savefig('pi1_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("✓ Figure saved: pi1_analysis.png")


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run benchmark
    results, N, C_tflops, M_gbs, P_gb = run_benchmark()
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save raw results
    df.to_csv('llm_inference_results.csv', index=False)
    print("✓ Data saved: llm_inference_results.csv")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Model parameters (N):     {N:,}")
    print(f"Configurations tested:    {len(df)}")
    print(f"Batch size range:         {df['batch_size'].min()} - {df['batch_size'].max()}")
    print(f"Sequence length range:    {df['seq_len'].min()} - {df['seq_len'].max()}")
    print(f"Latency range:            {df['latency_ms'].min():.2f} - {df['latency_ms'].max():.2f} ms")
    print(f"Throughput range:         {df['throughput_tokens_per_sec'].min():.1f} - {df['throughput_tokens_per_sec'].max():.1f} tok/s")
    print(f"π₁ (compute/memory) range: {df['Pi_1'].min():.3f} - {df['Pi_1'].max():.3f}")
    print(f"π₂ (BL/N) range:          {df['Pi_2'].min():.2e} - {df['Pi_2'].max():.2e}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(df, N, C_tflops, M_gbs, P_gb)
    
    print("\n" + "="*70)
    print("  BENCHMARK COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. llm_inference_results.csv  — Raw data")
    print("  2. results_figure.png         — Main visualization")
    print("  3. pi1_analysis.png (maybe)   — Additional analysis")
    print("\nThe result of the dimensional analysis.")
    print("="*70)
