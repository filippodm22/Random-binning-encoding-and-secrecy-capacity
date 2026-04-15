import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from task1 import WiretapUniformErrorChannel
from task2 import RandomBinningEncoder, HAMMING_CODE
from task3 import RandomBinningDecoder

# ──────────────────────────────────────────────
# Instantiate
# ──────────────────────────────────────────────
channel = WiretapUniformErrorChannel(n=7, r=1, s=3)
encoder = RandomBinningEncoder(HAMMING_CODE)
decoder = RandomBinningDecoder(HAMMING_CODE)

N_REALIZATIONS = 2**14
ALPHABET_SIZE  = 2**7   # 128
N_MESSAGES     = 8

# ──────────────────────────────────────────────
# Simulate encoder → eavesdropper channel
# ──────────────────────────────────────────────
u_samples = []   # message sent
z_samples = []   # eavesdropper output

for _ in range(N_REALIZATIONS):
    d_int = np.random.randint(N_MESSAGES)
    d_str = format(d_int, '03b')
    x     = encoder.encode(d_str)
    x_bits = tuple(int(b) for b in x)
    _, z_bits = channel.transmit(x_bits)
    z_int = int(''.join(str(b) for b in z_bits), 2)
    u_samples.append(d_int)
    z_samples.append(z_int)

u_samples = np.array(u_samples)
z_samples = np.array(z_samples)

# ──────────────────────────────────────────────
# 1. p_z|u(·|d) for each d
# ──────────────────────────────────────────────
p_z_given_u = np.zeros((N_MESSAGES, ALPHABET_SIZE))
for d_int in range(N_MESSAGES):
    mask = (u_samples == d_int)
    counts = np.bincount(z_samples[mask], minlength=ALPHABET_SIZE)
    p_z_given_u[d_int] = counts / counts.sum()

# ──────────────────────────────────────────────
# 2. Joint distribution p̂_u,z
# ──────────────────────────────────────────────
p_joint = np.zeros((N_MESSAGES, ALPHABET_SIZE))
for u, z in zip(u_samples, z_samples):
    p_joint[u, z] += 1
p_joint /= N_REALIZATIONS

# ──────────────────────────────────────────────
# 3. Marginals p̂_u and p̂_z
# ──────────────────────────────────────────────
p_u = p_joint.sum(axis=1)   # shape (8,)
p_z = p_joint.sum(axis=0)   # shape (128,)

print("Marginal p̂_u (should be ≈ 1/8 = 0.1250 for each message):")
for d_int in range(N_MESSAGES):
    print(f"  p̂_u({format(d_int,'03b')}) = {p_u[d_int]:.4f}")

# ──────────────────────────────────────────────
# 4. Empirical entropy H(u)
# ──────────────────────────────────────────────
H_u = -np.sum(p_u[p_u > 0] * np.log2(p_u[p_u > 0]))
print(f"\nEmpirical H(u) = {H_u:.4f} bits  (expected 3.0000 for uniform over 8 messages)")

# ──────────────────────────────────────────────
# 5. Empirical mutual information I(u;z)
# ──────────────────────────────────────────────
I_uz = 0.0
for d_int in range(N_MESSAGES):
    for c in range(ALPHABET_SIZE):
        puz = p_joint[d_int, c]
        if puz > 0 and p_u[d_int] > 0 and p_z[c] > 0:
            I_uz += puz * np.log2(puz / (p_u[d_int] * p_z[c]))

print(f"Empirical I(u;z)  = {I_uz:.6f} bits  (should be ≈ 0 for perfect secrecy)")

# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: p_z|u for each message (overlay)
ax = axes[0]
colors = plt.cm.tab10(np.linspace(0, 1, N_MESSAGES))
support_z = np.where(p_z > 0)[0]
for d_int in range(N_MESSAGES):
    ax.plot(support_z, p_z_given_u[d_int][support_z],
            marker='o', markersize=3, linewidth=0.8,
            label=f'd={format(d_int,"03b")}', color=colors[d_int], alpha=0.8)
ax.set_title(r'$\hat{p}_{z|u}(\cdot \mid d)$ for all $d \in \mathcal{M}$ — should overlap if perfect secrecy', fontsize=11)
ax.set_xlabel('z (integer representation)', fontsize=10)
ax.set_ylabel('Probability', fontsize=10)
ax.legend(loc='upper right', fontsize=7, ncol=4)

# Plot 2: marginal p̂_z
ax = axes[1]
ax.bar(support_z, p_z[support_z], width=0.6, color='steelblue', alpha=0.85)
ax.set_title(r'Marginal $\hat{p}_z(c)$', fontsize=11)
ax.set_xlabel('z (integer representation)', fontsize=10)
ax.set_ylabel('Probability', fontsize=10)

# Plot 3: marginal p̂_u
ax = axes[2]
ax.bar(range(N_MESSAGES), p_u, width=0.5, color='darkorange', alpha=0.85)
ax.axhline(1/8, color='red', linestyle='--', linewidth=1.2, label='Uniform 1/8')
ax.set_title(r'Marginal $\hat{p}_u(d)$', fontsize=11)
ax.set_xlabel('u (message integer)', fontsize=10)
ax.set_ylabel('Probability', fontsize=10)
ax.set_xticks(range(N_MESSAGES))
ax.set_xticklabels([format(d, '03b') for d in range(N_MESSAGES)])
ax.legend()

plt.tight_layout()
plt.savefig('task4_secrecy.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved.")