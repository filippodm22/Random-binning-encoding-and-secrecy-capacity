import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

from task2 import RandomBinningEncoder, HAMMING_CODE
from task3 import RandomBinningDecoder
from task8 import WiretapAWGN

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def h2(p: float) -> float:
    """Binary entropy function."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def secrecy_capacity_awgn(snr_b_db: float, snr_e_db: float) -> float:
    """
    Secrecy capacity of the Gaussian wiretap channel (real-valued AWGN).

        C_s = 0.5 · log2( (1 + SNR_B,lin) / (1 + SNR_E,lin) )   if SNR_B > SNR_E
        C_s = 0                                                  otherwise
    """
    if snr_b_db <= snr_e_db:
        return 0.0
    snr_b_lin = 10 ** (snr_b_db / 10.0)
    snr_e_lin = 10 ** (snr_e_db / 10.0)
    return 0.5 * np.log2((1 + snr_b_lin) / (1 + snr_e_lin))


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
    """Run the full chain encoder → AWGN wiretap channel → decoder for N messages."""
    messages = list(range(8))
    u_list, uhat_list, z_list = [], [], []

    # Vectorise: sample N messages and encode
    d_ints = np.random.choice(messages, size=N)
    x_seq  = np.zeros((N, 7), dtype=int)
    for i, d_int in enumerate(d_ints):
        d_str  = format(d_int, '03b')
        x_str  = encoder.encode(d_str)
        x_seq[i] = [int(b) for b in x_str]

    # Transmit through the AWGN wiretap channel (batch)
    Y, Z = channel.transmit_batch(x_seq)

    # Decode
    for i in range(N):
        y_str  = ''.join(map(str, Y[i]))
        uhat_str, _ = decoder.decode(y_str)
        u_list.append(int(d_ints[i]))
        uhat_list.append(int(uhat_str, 2))
        z_list.append(tuple(Z[i]))

    return np.array(u_list), np.array(uhat_list), z_list


def compute_distributions(u_arr, uhat_arr, z_list):
    """Same structure as Task 7: joint p(u,û,z), marginals, p(u,z)."""
    from collections import Counter

    N = len(u_arr)
    u_vals = list(range(8))

    joint_uuz = Counter(zip(u_arr, uhat_arr, z_list))
    p_uuz = {k: v / N for k, v in joint_uuz.items()}

    p_u    = {d: np.sum(u_arr   == d) / N for d in u_vals}
    p_uhat = {d: np.sum(uhat_arr == d) / N for d in u_vals}

    z_vals = list(set(z_list))
    p_z = {c: 0.0 for c in z_vals}
    for (_, _, zz), v in p_uuz.items():
        p_z[zz] += v

    joint_uz = Counter(zip(u_arr, z_list))
    p_uz = {k: v / N for k, v in joint_uz.items()}

    return p_u, p_uhat, p_z, p_uz, p_uuz


def total_variation_distance(p_uuz: dict, p_u: dict, p_z: dict) -> float:
    """d_V between empirical p(u,û,z) and ideal p_u(d)·𝟙{û=d}·p_z(c)."""
    all_keys = set(p_uuz.keys())
    for d in p_u:
        for c in p_z:
            all_keys.add((d, d, c))

    tv = 0.0
    for (d, dhat, c) in all_keys:
        p_emp   = p_uuz.get((d, dhat, c), 0.0)
        p_ideal = p_u.get(d, 0.0) * p_z.get(c, 0.0) if d == dhat else 0.0
        tv += abs(p_emp - p_ideal)
    return tv / 2.0


# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
np.random.seed(42)
N = 2**14

encoder = RandomBinningEncoder(HAMMING_CODE)
decoder = RandomBinningDecoder(HAMMING_CODE)

# Sweep parameters (dB). Range chosen so that:
#  - reliability sweep (vary SNR_B): covers from "too noisy to decode" up to
#    "essentially noiseless" for Bob;
#  - secrecy sweep (vary SNR_E):     covers from "Eve hears perfectly" down to
#    "Eve is very noisy" — mirrors the δ sweep in Task 7.
snr_b_values = np.linspace(10.0, 50.0, 21)   # reliability sweep
snr_e_values = np.linspace(50.0, 10.0, 21)   # secrecy sweep (decreasing = more noise for Eve)

FIXED_SNR_E = 15.0   # Eve is very noisy while we sweep Bob
FIXED_SNR_B = 45.0   # Bob is nearly noiseless while we sweep Eve

# ──────────────────────────────────────────────
# 1. RELIABILITY: P[û ≠ u] vs SNR_B
# ──────────────────────────────────────────────
print("Computing reliability sweep (P[û≠u] vs SNR_B) ...")
P_error_vs_snrb = []

for snr_b in snr_b_values:
    ch = WiretapAWGN(snr_b_db=snr_b, snr_e_db=FIXED_SNR_E, ell_x=7)
    u_arr, uhat_arr, _ = simulate(encoder, decoder, ch, N)
    P_err = np.mean(u_arr != uhat_arr)
    P_error_vs_snrb.append(P_err)
    print(f"  SNR_B={snr_b:5.1f} dB  P_err={P_err:.4f}")

# ──────────────────────────────────────────────
# 2. SECRECY: Î(u;z), TV distance, Pinsker bound, C_s vs SNR_E
# ──────────────────────────────────────────────
print("\nComputing secrecy sweep (I(u;z), TV, Cs vs SNR_E) ...")
I_uz_vs_snre = []
TV_vs_snre   = []
UB_vs_snre   = []
Cs_vs_snre   = []

for snr_e in snr_e_values:
    ch = WiretapAWGN(snr_b_db=FIXED_SNR_B, snr_e_db=snr_e, ell_x=7)
    u_arr, uhat_arr, z_list = simulate(encoder, decoder, ch, N)
    p_u, p_uhat, p_z, p_uz, p_uuz = compute_distributions(u_arr, uhat_arr, z_list)

    I_uz = empirical_mutual_info(p_uz, p_u, p_z)
    tv   = total_variation_distance(p_uuz, p_u, p_z)
    ub   = np.sqrt(np.log(2) / 2 * I_uz)               # Pinsker upper bound
    cs   = secrecy_capacity_awgn(FIXED_SNR_B, snr_e)

    I_uz_vs_snre.append(I_uz)
    TV_vs_snre.append(tv)
    UB_vs_snre.append(ub)
    Cs_vs_snre.append(cs)

    print(f"  SNR_E={snr_e:5.1f} dB  I(u;z)={I_uz:.4f}  TV={tv:.4f}  Cs={cs:.4f}")

# ──────────────────────────────────────────────
# PLOTS — same layout as Task 7 for immediate comparison
# ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Task 9 — System Security over Wiretap AWGN (PAM-128)\n'
             f'(Reliability: fixed SNR_E={FIXED_SNR_E} dB | '
             f'Secrecy: fixed SNR_B={FIXED_SNR_B} dB)',
             fontsize=13)

# Plot 1 — Reliability
ax = axes[0, 0]
ax.plot(snr_b_values, P_error_vs_snrb, 'o-', color='steelblue', linewidth=2, markersize=5)
ax.axvline(x=FIXED_SNR_E, color='red', linestyle='--', alpha=0.6,
           label=f'SNR_E={FIXED_SNR_E} dB (eavesdropper)')
ax.set_xlabel('SNR_B [dB] (legitimate channel)')
ax.set_ylabel('P[û ≠ u]')
ax.set_title('Reliability: Decoding Error Probability vs SNR_B')
ax.grid(alpha=0.3)
ax.legend()

# Plot 2 — Leaked information
ax = axes[0, 1]
ax.plot(snr_e_values, I_uz_vs_snre, 's-', color='tomato', linewidth=2, markersize=5,
        label='Empirical Î(u;z)')
ax.set_xlabel('SNR_E [dB] (eavesdropper channel)')
ax.set_ylabel('Î(u;z)  [bits]')
ax.set_title('Secrecy: Empirical Leaked Information vs SNR_E')
ax.invert_xaxis()   # high SNR_E (left) = Eve strong; low SNR_E (right) = Eve noisy
ax.grid(alpha=0.3)
ax.legend()

# Plot 3 — Unconditional security (TV vs Pinsker UB)
ax = axes[1, 0]
ax.plot(snr_e_values, TV_vs_snre, '^-', color='darkorange', linewidth=2, markersize=5,
        label='d_V(p_{uûz}, p*_{uûz})  [empirical]')
ax.plot(snr_e_values, UB_vs_snre, '--', color='purple', linewidth=2,
        label='Pinsker UB: √(ln2/2 · Î(u;z))')
ax.set_xlabel('SNR_E [dB] (eavesdropper channel)')
ax.set_ylabel('Total Variation Distance')
ax.set_title('Unconditional Security: TV Distance and Upper Bound')
ax.invert_xaxis()
ax.grid(alpha=0.3)
ax.legend()

# Plot 4 — Secrecy capacity
ax = axes[1, 1]
ax.plot(snr_e_values, Cs_vs_snre, 'D-', color='green', linewidth=2, markersize=5,
        label=f'C_s = ½·log₂((1+SNR_B)/(1+SNR_E))  [SNR_B={FIXED_SNR_B} dB]')
ax.axhline(y=3/7, color='red', linestyle='--', linewidth=1.5,
           label='Our rate R_s = 3/7 ≈ 0.429 bit/use')
ax.set_xlabel('SNR_E [dB] (eavesdropper channel)')
ax.set_ylabel('C_s  [bits/channel use]')
ax.set_title('Secrecy Capacity of the Wiretap AWGN Channel')
ax.invert_xaxis()
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('task9_security_awgn.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved → task9_security_awgn.png")

# ──────────────────────────────────────────────
# Printout summary
# ──────────────────────────────────────────────
print("\n══════════════════════════════════════════════════════")
print("   SUMMARY")
print("══════════════════════════════════════════════════════")
print(f"  System secret rate                   : 3/7 ≈ {3/7:.4f} bits/use")
print(f"  Theoretical C_s at (SNR_B={FIXED_SNR_B}, SNR_E={FIXED_SNR_E}) dB : "
      f"{secrecy_capacity_awgn(FIXED_SNR_B, FIXED_SNR_E):.4f} bits/use")
print(f"  Min P_err (SNR_B→{snr_b_values[-1]:.0f} dB)           : {min(P_error_vs_snrb):.5f}")
print(f"  Max P_err (SNR_B→{snr_b_values[0]:.0f} dB)           : {max(P_error_vs_snrb):.5f}")
print(f"  I(u;z) at SNR_E={snr_e_values[0]:.0f} dB (Eve strong): {I_uz_vs_snre[0]:.5f} bits")
print(f"  I(u;z) at SNR_E={snr_e_values[-1]:.0f} dB (Eve noisy) : {I_uz_vs_snre[-1]:.5f} bits")
print("══════════════════════════════════════════════════════")
