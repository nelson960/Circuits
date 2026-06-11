#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Directory to save figures
ARTIFACT_DIR = Path("/Users/nelson/.gemini/antigravity/brain/a825cb9d-d6db-43d7-9266-14b61cc53bc3")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def get_orthonormal_basis(d: int, r: int) -> torch.Tensor:
    """Generates an orthonormal basis of shape (d, r) using QR decomposition."""
    X = torch.randn(d, r, dtype=torch.double)
    Q, _ = torch.linalg.qr(X)
    return Q

def generate_subspaces(d: int, r_old: int, r_new: int, overlap: float, mode: str = "discrete"):
    """
    Generates U_old (d, r_old) and V_new (d, r_new) with a controlled overlap.
    - discrete: V_new shares exactly k basis vectors with U_old, rest are orthogonal.
    - continuous: V_new is rotated towards U_old by an angle theta.
    """
    Q = get_orthonormal_basis(d, r_old + r_new)
    U_old = Q[:, :r_old]
    
    if mode == "discrete":
        k = int(round(overlap * r_new))
        # Share the last k vectors of U_old, and take the rest from orthogonal complement
        V_shared = U_old[:, r_old - k:]
        V_ortho = Q[:, r_old:r_old + r_new - k]
        V_new = torch.cat([V_shared, V_ortho], dim=1)
    elif mode == "continuous":
        # Let's align V_new column-by-column with U_old with angle theta
        theta = np.arccos(np.sqrt(overlap))
        W = Q[:, r_old:r_old + r_new]
        V_new = np.cos(theta) * U_old[:, :r_new] + np.sin(theta) * W
    else:
        raise ValueError(f"Unknown mode: {mode}")
        
    return U_old, V_new

def run_experiment_1():
    print("--- Experiment 1: Subspace Overlap Sweep (Pure Math) ---")
    d = 100
    r_old = 20
    r_new = 20
    overlaps = [0.0, 0.25, 0.50, 0.75, 1.0]
    
    for mode in ["discrete", "continuous"]:
        results = []
        for overlap in overlaps:
            U_old, V_new = generate_subspaces(d, r_old, r_new, overlap, mode=mode)
            
            # Create a raw gradient G
            d_out = 100
            G = torch.randn(d_out, d, dtype=torch.double)
            
            # Project to find G_safe
            P_free = torch.eye(d, dtype=torch.double) - U_old @ U_old.T
            G_safe = G @ P_free
            
            # Measurements
            old_interference = torch.norm(G_safe @ U_old, p="fro").item()
            new_learning_power = torch.norm(G_safe @ V_new, p="fro").item() / torch.norm(G @ V_new, p="fro").item()
            safe_update_ratio = torch.norm(G_safe, p="fro").item() / torch.norm(G, p="fro").item()
            
            results.append({
                "overlap": overlap,
                "old_interference": old_interference,
                "new_learning_power": new_learning_power,
                "safe_update_ratio": safe_update_ratio
            })
            
            # Print results
            print(f"[{mode.upper()}] Overlap: {overlap:4.0%} | Old Int: {old_interference:11.3e} | New LP: {new_learning_power:.4f} | Update Ratio: {safe_update_ratio:.4f}")
            
            # Safety assertion
            assert old_interference < 1e-12, f"Old interference is too high: {old_interference}"
            
        # Plot
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        color = 'tab:red'
        ax1.set_xlabel('Old-New Subspace Overlap')
        ax1.set_ylabel('Old Interference (Frobenius Norm)', color=color)
        ax1.plot(overlaps, [r["old_interference"] for r in results], 'o--', color=color, label='Old Interference')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(-1e-14, 1e-13)
        
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('New Learning Power / Safe Update Ratio', color=color)
        ax2.plot(overlaps, [r["new_learning_power"] for r in results], 's-', color=color, label='New Learning Power')
        ax2.plot(overlaps, [r["safe_update_ratio"] for r in results], '^:', color='tab:green', label='Safe Update Ratio')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(-0.05, 1.05)
        
        plt.title(f"Subspace Overlap vs. Learning Metrics ({mode.capitalize()} Overlap)")
        fig.tight_layout()
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left')
        
        plot_path = ARTIFACT_DIR / f"geometric_preservation_{mode}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot to {plot_path}")
    print()

def run_experiment_2():
    print("--- Experiment 2: Linear Output & Margin Preservation ---")
    d = 100
    r_old = 20
    d_out = 50
    
    # 1. Setup linear model W
    W = torch.randn(d_out, d, dtype=torch.double)
    
    # 2. Get old subspace and inputs
    U_old = get_orthonormal_basis(d, r_old)
    N_old = 100
    A = torch.randn(r_old, N_old, dtype=torch.double)
    X_old = U_old @ A  # Old inputs are in span(U_old)
    
    # Target output
    Y_old = W @ X_old
    
    # 3. Create raw gradient and project
    G = torch.randn(d_out, d, dtype=torch.double)
    P_free = torch.eye(d, dtype=torch.double) - U_old @ U_old.T
    G_safe = G @ P_free
    
    # 4. Updates
    eta = 0.1
    W_safe = W - eta * G_safe
    W_naive = W - eta * G
    
    # 5. Output checks
    Y_safe = W_safe @ X_old
    Y_naive = W_naive @ X_old
    
    safe_error = torch.norm(Y_safe - Y_old, p="fro").item()
    naive_error = torch.norm(Y_naive - Y_old, p="fro").item()
    
    print(f"Safe Update output error:  {safe_error:.3e}")
    print(f"Naive Update output error: {naive_error:.4f}")
    assert safe_error < 1e-12, f"Safe update output error is too high: {safe_error}"
    
    # 6. Semantic Margin Preservation
    # Define margin = logit(correct) - logit(wrong) for a 10-class output
    num_classes = 10
    W_cls = W[:num_classes, :]
    W_cls_safe = W_safe[:num_classes, :]
    W_cls_naive = W_naive[:num_classes, :]
    
    # Randomly assign correct and wrong classes for N_old samples
    correct_classes = torch.randint(0, num_classes, (N_old,))
    wrong_classes = (correct_classes + torch.randint(1, num_classes, (N_old,))) % num_classes
    
    logits_old = W_cls @ X_old
    logits_safe = W_cls_safe @ X_old
    logits_naive = W_cls_naive @ X_old
    
    margins_old = logits_old[correct_classes, torch.arange(N_old)] - logits_old[wrong_classes, torch.arange(N_old)]
    margins_safe = logits_safe[correct_classes, torch.arange(N_old)] - logits_safe[wrong_classes, torch.arange(N_old)]
    margins_naive = logits_naive[correct_classes, torch.arange(N_old)] - logits_naive[wrong_classes, torch.arange(N_old)]
    
    margin_diff_safe = torch.max(torch.abs(margins_safe - margins_old)).item()
    margin_diff_naive = torch.max(torch.abs(margins_naive - margins_old)).item()
    
    print(f"Safe Update max margin diff:  {margin_diff_safe:.3e}")
    print(f"Naive Update max margin diff: {margin_diff_naive:.4f}")
    assert margin_diff_safe < 1e-12, f"Safe update margin difference is too high: {margin_diff_safe}"
    print()

def run_experiment_3():
    print("--- Experiment 3: MLP Subspace Preservation ---")
    d_in = 100
    d_hidden = 64
    d_out = 10
    r_old = 20
    
    # 1. Create 2-Layer MLP parameters
    W_in = torch.randn(d_hidden, d_in, dtype=torch.double)
    b_in = torch.randn(d_hidden, 1, dtype=torch.double)
    W_out = torch.randn(d_out, d_hidden, dtype=torch.double)
    b_out = torch.randn(d_out, 1, dtype=torch.double)
    
    def mlp_forward(x, W_i, b_i, W_o, b_o):
        # x is (d_in, N)
        h = torch.tanh(W_i @ x + b_i)  # non-linear activation (tanh)
        y = W_o @ h + b_o
        return h, y
        
    # Generate old inputs
    U_old = get_orthonormal_basis(d_in, r_old)
    N_old = 100
    A = torch.randn(r_old, N_old, dtype=torch.double)
    X_old = U_old @ A
    
    # Baseline outputs
    H_old, Y_old = mlp_forward(X_old, W_in, b_in, W_out, b_out)
    
    # SCENARIO A: Update Input Weights (W_in)
    G_in = torch.randn_like(W_in)
    # Project: G_in_safe = G_in @ (I - U_old @ U_old^T)
    P_free = torch.eye(d_in, dtype=torch.double) - U_old @ U_old.T
    G_in_safe = G_in @ P_free
    
    W_in_safe = W_in - 0.1 * G_in_safe
    H_safe_in, Y_safe_in = mlp_forward(X_old, W_in_safe, b_in, W_out, b_out)
    
    # SCENARIO B: Update Output Weights (W_out)
    # First, we need to find the subspace occupied by H_old (shape: d_hidden, N_old)
    # Perform SVD to get orthonormal basis for H_old
    U_h, _, _ = torch.linalg.svd(H_old, full_matrices=False)
    # Since H_old has rank at most min(d_hidden, N_old), U_h spans the hidden activations
    # Project G_out using U_h
    G_out = torch.randn_like(W_out)
    P_free_h = torch.eye(d_hidden, dtype=torch.double) - U_h @ U_h.T
    G_out_safe = G_out @ P_free_h
    
    W_out_safe = W_out - 0.1 * G_out_safe
    _, Y_safe_out = mlp_forward(X_old, W_in, b_in, W_out_safe, b_out)
    
    # NAIVE Update of both for comparison
    W_in_naive = W_in - 0.1 * G_in
    W_out_naive = W_out - 0.1 * G_out
    _, Y_naive = mlp_forward(X_old, W_in_naive, b_in, W_out_naive, b_out)
    
    # Calculations
    in_update_error = torch.max(torch.abs(Y_safe_in - Y_old)).item()
    out_update_error = torch.max(torch.abs(Y_safe_out - Y_old)).item()
    naive_error = torch.max(torch.abs(Y_naive - Y_old)).item()
    
    print(f"MLP Input-Weight Safe Update output error:  {in_update_error:.3e}")
    print(f"MLP Output-Weight Safe Update output error: {out_update_error:.3e}")
    print(f"MLP Naive Update output error:               {naive_error:.4f}")
    
    assert in_update_error < 1e-12, f"MLP Input-Weight safe update error is too high: {in_update_error}"
    assert out_update_error < 1e-12, f"MLP Output-Weight safe update error is too high: {out_update_error}"
    print()

def run_experiment_4():
    print("--- Experiment 4: Transformer MLP Block Test ---")
    d_model = 128
    d_ff = 256
    r_old = 20
    N_old = 100
    
    # 1. Transformer MLP Block weights
    W_gate = torch.randn(d_ff, d_model, dtype=torch.double)
    W_up = torch.randn(d_ff, d_model, dtype=torch.double)
    W_down = torch.randn(d_model, d_ff, dtype=torch.double)
    
    def transformer_mlp_forward(x, W_g, W_u, W_d):
        # x is (d_model, N)
        # Standard Gated MLP (e.g. SwiGLU / GeGLU style)
        # we'll use gelu(x_gate) * x_up
        x_gate = W_g @ x
        x_up = W_u @ x
        h = torch.nn.functional.gelu(x_gate) * x_up  # (d_ff, N)
        mlp_out = W_d @ h
        # Residual output
        return h, x + mlp_out
        
    # Generate old residual stream inputs
    U_old = get_orthonormal_basis(d_model, r_old)
    A = torch.randn(r_old, N_old, dtype=torch.double)
    X_old = U_old @ A
    
    # Base outputs
    H_old, Y_old = transformer_mlp_forward(X_old, W_gate, W_up, W_down)
    
    # SCENARIO A: Protect Input Weights (W_gate and W_up)
    G_gate = torch.randn_like(W_gate)
    G_up = torch.randn_like(W_up)
    
    P_free = torch.eye(d_model, dtype=torch.double) - U_old @ U_old.T
    G_gate_safe = G_gate @ P_free
    G_up_safe = G_up @ P_free
    
    W_gate_safe = W_gate - 0.1 * G_gate_safe
    W_up_safe = W_up - 0.1 * G_up_safe
    
    _, Y_safe_in = transformer_mlp_forward(X_old, W_gate_safe, W_up_safe, W_down)
    
    # SCENARIO B: Protect Down Weight (W_down)
    # Find the hidden subspace of h_old (d_ff, N_old)
    U_h, _, _ = torch.linalg.svd(H_old, full_matrices=False)
    
    G_down = torch.randn_like(W_down)
    P_free_h = torch.eye(d_ff, dtype=torch.double) - U_h @ U_h.T
    G_down_safe = G_down @ P_free_h
    
    W_down_safe = W_down - 0.1 * G_down_safe
    _, Y_safe_down = transformer_mlp_forward(X_old, W_gate, W_up, W_down_safe)
    
    # NAIVE Update for comparison
    W_gate_naive = W_gate - 0.1 * G_gate
    W_up_naive = W_up - 0.1 * G_up
    W_down_naive = W_down - 0.1 * G_down
    _, Y_naive = transformer_mlp_forward(X_old, W_gate_naive, W_up_naive, W_down_naive)
    
    # Errors
    in_error = torch.max(torch.abs(Y_safe_in - Y_old)).item()
    down_error = torch.max(torch.abs(Y_safe_down - Y_old)).item()
    naive_error = torch.max(torch.abs(Y_naive - Y_old)).item()
    
    print(f"Transformer MLP Input-Weight Safe Update output error: {in_error:.3e}")
    print(f"Transformer MLP Down-Weight Safe Update output error:  {down_error:.3e}")
    print(f"Transformer MLP Naive Update output error:              {naive_error:.4f}")
    
    assert in_error < 1e-11, f"Transformer MLP input-weight safe update error is too high: {in_error}"
    assert down_error < 1e-11, f"Transformer MLP down-weight safe update error is too high: {down_error}"
    print("All mathematical simulations and tests completed successfully.")

if __name__ == "__main__":
    run_experiment_1()
    run_experiment_2()
    run_experiment_3()
    run_experiment_4()
