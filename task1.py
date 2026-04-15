import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

def int_to_bits(n, length=7):
    return tuple(int(b) for b in format(n, f'0{length}b'))

def bits_to_int(bits):
    return int(''.join(str(b) for b in bits), 2)

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

def xor_bits(a, b):
    return tuple(x ^ y for x, y in zip(a, b))

def error_vectors(n, max_errors):
    vectors = []
    for weight in range(max_errors + 1):
        for positions in combinations(range(n), weight):
            e = [0] * n
            for p in positions:
                e[p] = 1
            vectors.append(tuple(e))
    return vectors

class WiretapUniformErrorChannel:
    def __init__(self, n=7, r=1, s=3):
        self.n = n
        self.r = r
        self.s = s
        self.legit_errors = error_vectors(n, r)
        self.eave_errors  = error_vectors(n, s)
        print(f"Legit error vectors (weight <= {r}): {len(self.legit_errors)}")
        print(f"Eave  error vectors (weight <= {s}): {len(self.eave_errors)}")

    def transmit(self, x_bits):
        e_y = self.legit_errors[np.random.randint(len(self.legit_errors))]
        e_z = self.eave_errors [np.random.randint(len(self.eave_errors))]
        return xor_bits(x_bits, e_y), xor_bits(x_bits, e_z)

    def simulate(self, x_bits, n_realizations):
        y_list, z_list = [], []
        for _ in range(n_realizations):
            y, z = self.transmit(x_bits)
            y_list.append(bits_to_int(y))
            z_list.append(bits_to_int(z))
        return np.array(y_list), np.array(z_list)

def empirical_pmf(samples, alphabet_size):
    counts = np.bincount(samples, minlength=alphabet_size)
    return counts / counts.sum()

def mutual_information(p_joint, p_y, p_z):
    mi = 0.0
    for yi in range(len(p_y)):
        for zi in range(len(p_z)):
            pyz = p_joint[yi, zi]
            if pyz > 0 and p_y[yi] > 0 and p_z[zi] > 0:
                mi += pyz * np.log2(pyz / (p_y[yi] * p_z[zi]))
    return mi

if __name__ == '__main__':
    N_REALIZATIONS = 2**14
    ALPHABET_SIZE  = 2**7
    N_BITS         = 7
    R, S           = 1, 3

    x_str  = "1001000"
    x_bits = tuple(int(b) for b in x_str)
    x_int  = bits_to_int(x_bits)

    print(f"Input x = {x_str}  (int {x_int})")
    print(f"Simulating {N_REALIZATIONS} realizations...\n")

    channel = WiretapUniformErrorChannel(n=N_BITS, r=R, s=S)
    y_samples, z_samples = channel.simulate(x_bits, N_REALIZATIONS)

    p_y = empirical_pmf(y_samples, ALPHABET_SIZE)
    p_z = empirical_pmf(z_samples, ALPHABET_SIZE)

    support_y = np.where(p_y > 0)[0]
    support_z = np.where(p_z > 0)[0]
    print(f"Support size of p̂_y|x : {len(support_y)}  (expected {len(channel.legit_errors)})")
    print(f"Support size of p̂_z|x : {len(support_z)}  (expected {len(channel.eave_errors)})")

    p_y_nonzero = p_y[support_y]
    p_z_nonzero = p_z[support_z]
    print(f"\np̂_y|x values (should all ≈ 1/8  = {1/8:.4f}):")
    print(f"  min={p_y_nonzero.min():.4f}  max={p_y_nonzero.max():.4f}  mean={p_y_nonzero.mean():.4f}")
    print(f"p̂_z|x values (should all ≈ 1/64 = {1/64:.4f}):")
    print(f"  min={p_z_nonzero.min():.4f}  max={p_z_nonzero.max():.4f}  mean={p_z_nonzero.mean():.4f}")

    p_joint = np.zeros((ALPHABET_SIZE, ALPHABET_SIZE))
    for yi, zi in zip(y_samples, z_samples):
        p_joint[yi, zi] += 1
    p_joint /= N_REALIZATIONS

    mi = mutual_information(p_joint, p_y, p_z)
    print(f"\nEmpirical I(y;z|x) = {mi:.6f} bits  (should be ≈ 0 if y⊥z|x)")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    ax = axes[0]
    ax.bar(support_y, p_y[support_y], width=0.6, color='steelblue', alpha=0.85)
    ax.axhline(1/8, color='red', linestyle='--', linewidth=1.2, label='Uniform 1/8')
    ax.set_title(r'Empirical distribution $\hat{p}_{y|x}(\cdot \mid 1001000)$ — Legitimate channel (r=1)', fontsize=12)
    ax.set_xlabel('y (integer representation)', fontsize=10)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_xlim(-1, ALPHABET_SIZE)
    ax.legend()
    for yi in support_y:
        ax.annotate(format(yi, '07b'), xy=(yi, p_y[yi]),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=6, rotation=90)
    ax = axes[1]
    ax.bar(support_z, p_z[support_z], width=0.6, color='darkorange', alpha=0.85)
    ax.axhline(1/64, color='red', linestyle='--', linewidth=1.2, label='Uniform 1/64')
    ax.set_title(r'Empirical distribution $\hat{p}_{z|x}(\cdot \mid 1001000)$ — Eavesdropper channel (s=3)', fontsize=12)
    ax.set_xlabel('z (integer representation)', fontsize=10)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_xlim(-1, ALPHABET_SIZE)
    ax.legend()
    plt.tight_layout()
    plt.savefig('task1_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved.")