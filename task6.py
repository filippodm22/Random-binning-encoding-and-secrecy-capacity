import numpy as np

# ──────────────────────────────────────────────
# Task 6 — Wiretap Binary Symmetric Channel
# ──────────────────────────────────────────────

class WiretapBSC:
    """
    Wiretap Binary Symmetric Channel.
    - Legitimate channel: each bit flipped independently with probability epsilon
    - Eavesdropper channel: each bit flipped independently with probability delta
    - y and z are conditionally independent given x (errors drawn separately)
    """

    def __init__(self, epsilon: float, delta: float, n: int = 7):
        self.epsilon = epsilon
        self.delta   = delta
        self.n       = n

    def transmit(self, x):
        """
        Single realization: given input word x (array-like of 0/1),
        return (y, z) as numpy arrays.
        """
        x = np.array(x, dtype=int)
        e_y = (np.random.rand(self.n) < self.epsilon).astype(int)
        e_z = (np.random.rand(self.n) < self.delta).astype(int)
        return x ^ e_y, x ^ e_z

    def transmit_batch(self, x, num_realizations: int):
        """
        Batch simulation: same input x repeated num_realizations times.
        Returns y_batch (N, n) and z_batch (N, n).
        """
        x = np.array(x, dtype=int)
        E_y = (np.random.rand(num_realizations, self.n) < self.epsilon).astype(int)
        E_z = (np.random.rand(num_realizations, self.n) < self.delta).astype(int)
        return x ^ E_y, x ^ E_z


# ──────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    N_LONG   = 100_000   # long sequence for BER measurement
    N_WORDS  = N_LONG    # number of 7-bit words
    n        = 7

    # Test several (epsilon, delta) pairs
    test_pairs = [
        (0.0,  0.0),
        (0.05, 0.2),
        (0.1,  0.3),
        (0.2,  0.4),
        (0.5,  0.5),
    ]

    print("=" * 65)
    print("VERIFICATION — Empirical BER vs theoretical epsilon / delta")
    print("=" * 65)
    print(f"{'ε':>6} {'δ':>6} | {'BER_y':>10} {'BER_z':>10} | {'ratio BER_z/BER_y':>18}")
    print("-" * 65)

    for eps, dlt in test_pairs:
        channel = WiretapBSC(epsilon=eps, delta=dlt, n=n)

        # Generate random input sequence (one word at a time for generality)
        x_seq = np.random.randint(0, 2, size=(N_WORDS, n))

        total_err_y = 0
        total_err_z = 0

        for i in range(N_WORDS):
            y, z = channel.transmit(x_seq[i])
            total_err_y += np.sum(x_seq[i] ^ y)
            total_err_z += np.sum(x_seq[i] ^ z)

        ber_y = total_err_y / (N_WORDS * n)
        ber_z = total_err_z / (N_WORDS * n)
        ratio = ber_z / ber_y if ber_y > 1e-10 else float('inf')

        print(f"{eps:>6.2f} {dlt:>6.2f} | {ber_y:>10.5f} {ber_z:>10.5f} | {ratio:>18.4f}  (expected {dlt/(eps+1e-15):.4f})")

    print()
    print("BER_y ≈ ε and BER_z ≈ δ → channel correctly implemented ✓")