import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from task2 import RandomBinningEncoder, HAMMING_CODE
from task3 import RandomBinningDecoder
from task6 import WiretapBSC

# Helpers

def h2(p: float) -> float:
    """Binary entropy function."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def secrecy_capacity_bsc(eps: float, dlt: float) -> float:
    """Theoretical secrecy capacity of the wiretap BSC."""
    if abs(eps - 0.5) <= abs(dlt - 0.5):
        return 0.0
    return h2(dlt) - h2(eps)

def empirical_mutual_info(p_uz: dict, p_u: dict, p_z: dict) -> float:
    """I(u;z) = Σ p(u,z) log2[ p(u,z) / (p(u)·p(z)) ]"""
    I = 0.0
    eps = 1e-15
    for (d, c), p_joint in p_uz.items():
        if p_joint > eps:
            denom = p_u.get(d, eps) * p_z.get(c, eps)
            I += p_joint * np.log2(p_joint / denom)
    return max(I, 0.0)

def simulate(encoder, decoder, channel, N=2**14):

    messages = list(range(8))
    u_list, uhat_list, z_list = [], [], []

    for _ in range(N):
        d_int  = np.random.choice(messages)
        d_str  = format(d_int, '03b')

        # Encode
        x_str  = encoder.encode(d_str)
        x_bits = np.array([int(b) for b in x_str])

        # Transmit
        y_bits, z_bits = channel.transmit(x_bits)
        y_str = ''.join(map(str, y_bits))

        # Decode
        u_hat_str, _ = decoder.decode(y_str)

        u_list.append(d_int)
        uhat_list.append(int(u_hat_str, 2))
        z_list.append(tuple(z_bits))

    return np.array(u_list), np.array(uhat_list), z_list


def compute_distributions(u_arr, uhat_arr, z_list):

    N = len(u_arr)
    u_vals   = list(range(8))
    z_vals   = list(set(z_list))

    # Joint p(u, û, z)
    from collections import Counter
    joint_uuz = Counter(zip(u_arr, uhat_arr, z_list))
    p_uuz = {k: v / N for k, v in joint_uuz.items()}

    # Marginals
    p_u    = {d: np.sum(u_arr == d) / N for d in u_vals}
    p_uhat = {d: np.sum(uhat_arr == d) / N for d in u_vals}
    p_z    = {}
    for c in z_vals:
        p_z[c] = sum(v for (_, _, zz), v in p_uuz.items() if zz == c)

    # Joint p(u, z)
    joint_uz = Counter(zip(u_arr, z_list))
    p_uz = {k: v / N for k, v in joint_uz.items()}

    return p_u, p_uhat, p_z, p_uz, p_uuz


def total_variation_distance(p_uuz: dict, p_u: dict, p_z: dict) -> float:

    # Collect all (u, û, z) combinations seen
    all_keys = set(p_uuz.keys())
    # Add ideal keys (u=û combinations)
    for d in p_u:
        for c in p_z:
            all_keys.add((d, d, c))

    tv = 0.0
    for (d, dhat, c) in all_keys:
        p_emp  = p_uuz.get((d, dhat, c), 0.0)
        p_ideal = p_u.get(d, 0.0) * p_z.get(c, 0.0) if d == dhat else 0.0
        tv += abs(p_emp - p_ideal)

    return tv / 2.0

# Setup

np.random.seed(42)
N = 2**14

encoder = RandomBinningEncoder(HAMMING_CODE)
decoder = RandomBinningDecoder(HAMMING_CODE)

# Sweep parameters 
eps_values = np.linspace(0.0, 0.48, 25)   # vary ε
dlt_values = np.linspace(0.0, 0.48, 25)   # vary δ

FIXED_DELTA = 0.35   # fixed δ for reliability sweep
FIXED_EPS   = 0.05   # fixed ε for secrecy sweep

# 1. RELIABILITY: P[û ≠ u] vs ε
print("Computing reliability sweep (P[û≠u] vs ε) ...")
P_error_vs_eps = []

for eps in eps_values:
    ch = WiretapBSC(epsilon=eps, delta=FIXED_DELTA, n=7)
    u_arr, uhat_arr, z_list = simulate(encoder, decoder, ch, N)
    P_err = np.mean(u_arr != uhat_arr)
    P_error_vs_eps.append(P_err)
    print(f"  ε={eps:.3f}  P_err={P_err:.4f}")

# 2. SECRECY: Î(u;z) vs δ
print("\nComputing secrecy sweep (I(u;z) vs δ) ...")
I_uz_vs_dlt     = []
TV_vs_dlt       = []
UB_vs_dlt       = []
Cs_vs_dlt       = []

for dlt in dlt_values:
    ch = WiretapBSC(epsilon=FIXED_EPS, delta=dlt, n=7)
    u_arr, uhat_arr, z_list = simulate(encoder, decoder, ch, N)
    p_u, p_uhat, p_z, p_uz, p_uuz = compute_distributions(u_arr, uhat_arr, z_list)

    I_uz  = empirical_mutual_info(p_uz, p_u, p_z)
    tv    = total_variation_distance(p_uuz, p_u, p_z)
    ub    = np.sqrt(np.log(2) / 2 * I_uz)   # Pinsker upper bound
    cs    = secrecy_capacity_bsc(FIXED_EPS, dlt)

    I_uz_vs_dlt.append(I_uz)
    TV_vs_dlt.append(tv)
    UB_vs_dlt.append(ub)
    Cs_vs_dlt.append(cs)

    print(f"  δ={dlt:.3f}  I(u;z)={I_uz:.4f}  TV={tv:.4f}  Cs={cs:.4f}")

# PLOTS

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Task 7 — System Security over Wiretap BSC\n'
             f'(Reliability: fixed δ={FIXED_DELTA} | Secrecy: fixed ε={FIXED_EPS})',
             fontsize=13)

# Plot 1: Reliability
ax = axes[0, 0]
ax.plot(eps_values, P_error_vs_eps, 'o-', color='steelblue', linewidth=2, markersize=5)
ax.axvline(x=FIXED_DELTA, color='red', linestyle='--', alpha=0.6, label=f'δ={FIXED_DELTA} (eavesdropper)')
ax.set_xlabel('ε (legitimate channel crossover probability)')
ax.set_ylabel('P[û ≠ u]')
ax.set_title('Reliability: Decoding Error Probability vs ε')
ax.grid(alpha=0.3)
ax.legend()

# Plot 2: Leaked Information
ax = axes[0, 1]
ax.plot(dlt_values, I_uz_vs_dlt, 's-', color='tomato', linewidth=2, markersize=5,
        label='Empirical Î(u;z)')
ax.set_xlabel('δ (eavesdropper crossover probability)')
ax.set_ylabel('Î(u;z)  [bits]')
ax.set_title('Secrecy: Empirical Leaked Information vs δ')
ax.grid(alpha=0.3)
ax.legend()

# Plot 3: Upper bound on unconditional security
ax = axes[1, 0]
ax.plot(dlt_values, TV_vs_dlt, '^-', color='darkorange', linewidth=2, markersize=5,
        label='d_V(p_{uûz}, p*_{uûz})  [empirical]')
ax.plot(dlt_values, UB_vs_dlt, '--', color='purple', linewidth=2,
        label='Pinsker UB: √(ln2/2 · Î(u;z))')
ax.set_xlabel('δ (eavesdropper crossover probability)')
ax.set_ylabel('Total Variation Distance')
ax.set_title('Unconditional Security: TV Distance and Upper Bound')
ax.grid(alpha=0.3)
ax.legend()

# Plot 4: Secrecy Capacity
ax = axes[1, 1]
ax.plot(dlt_values, Cs_vs_dlt, 'D-', color='green', linewidth=2, markersize=5,
        label=f'C_s = max(0, h₂(δ)−h₂(ε={FIXED_EPS}))')
ax.axhline(y=3/7, color='red', linestyle='--', linewidth=1.5,
           label='Our rate R_s = 3/7 ≈ 0.429 bit/use')
ax.set_xlabel('δ (eavesdropper crossover probability)')
ax.set_ylabel('C_s  [bits/channel use]')
ax.set_title('Secrecy Capacity of the Wiretap BSC')
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('task7_security_bsc.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved → task7_security_bsc.png")

# Printout

print("\n══════════════════════════════════════════════════════")
print("   SUMMARY")
print("══════════════════════════════════════════════════════")
print(f"  System secret rate          : 3/7 ≈ {3/7:.4f} bits/use")
print(f"  Theoretical C_s at (ε={FIXED_EPS}, δ={FIXED_DELTA}): "
      f"{secrecy_capacity_bsc(FIXED_EPS, FIXED_DELTA):.4f} bits/use")
print(f"  Min P_err (ε→0)             : {min(P_error_vs_eps):.5f}")
print(f"  I(u;z) at δ=0               : {I_uz_vs_dlt[0]:.5f} bits  (should ≈ 0)")
print(f"  I(u;z) at δ={dlt_values[-1]:.2f}           : {I_uz_vs_dlt[-1]:.5f} bits")
print("══════════════════════════════════════════════════════")
