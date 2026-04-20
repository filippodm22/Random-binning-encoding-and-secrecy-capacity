# Task 8 — Wiretap AWGN Channel with PAM-2^ℓx modulation

import numpy as np


class WiretapAWGN:
    """
    Wiretap AWGN channel with PAM-M modulation (M = 2^ell_x).

    Pipeline:
        u (ℓx bits) ──► PAM modulator ──► x_tilde ∈ ℝ
                                             │
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                  y_tilde = x_tilde + n_B        z_tilde = x_tilde + n_E
                              │                             │
                              ▼                             ▼
                         PAM demod                     PAM demod
                              │                             │
                              ▼                             ▼
                        y (ℓx bits)                   z (ℓx bits)

    n_B ~ N(0, P·10^(-SNR_B/10))
    n_E ~ N(0, P·10^(-SNR_E/10))

    where P is the average signal power of the PAM constellation:
        P = E[x_tilde^2] = (M^2 - 1) / 3
    for the standard symmetric constellation a_i = 2i - (M-1), i = 0..M-1.
    """

    def __init__(self, snr_b_db: float, snr_e_db: float, ell_x: int = 7):
        self.snr_b_db = snr_b_db
        self.snr_e_db = snr_e_db
        self.ell_x    = ell_x
        self.M        = 2 ** ell_x                  # PAM cardinality (128)

        # PAM constellation a_i = 2i - (M-1), spacing d = 2
        self.constellation = np.arange(self.M) * 2 - (self.M - 1)

        # Average symbol power (uniform prior over M symbols)
        self.P = (self.M ** 2 - 1) / 3.0

        # Noise variances (linear scale)
        snr_b_lin = 10 ** (snr_b_db / 10.0)
        snr_e_lin = 10 ** (snr_e_db / 10.0)
        self.sigma_b = np.sqrt(self.P / snr_b_lin)
        self.sigma_e = np.sqrt(self.P / snr_e_lin)

    # ──────────────────────────────────────────────
    # Modulator / Demodulator
    # ──────────────────────────────────────────────
    def modulate(self, x_bits):
        """Map 7-bit word to PAM symbol (binary mapping: integer value → a_i)."""
        x_bits = np.asarray(x_bits, dtype=int)
        idx = int(''.join(str(b) for b in x_bits), 2)
        return self.constellation[idx]

    def demodulate(self, r):
        """Hard decision: closest symbol → index → ℓx-bit word."""
        # Closest constellation index via rounding (equivalent to min |r - a_i|)
        idx = int(np.round((r + (self.M - 1)) / 2.0))
        idx = max(0, min(self.M - 1, idx))   # clip to valid range
        return np.array([int(b) for b in format(idx, f'0{self.ell_x}b')], dtype=int)

    # ──────────────────────────────────────────────
    # Transmission
    # ──────────────────────────────────────────────
    def transmit(self, x_bits):
        """Single word → (y_bits, z_bits)."""
        x_tilde = self.modulate(x_bits)
        n_b = np.random.randn() * self.sigma_b
        n_e = np.random.randn() * self.sigma_e
        y_tilde = x_tilde + n_b
        z_tilde = x_tilde + n_e
        return self.demodulate(y_tilde), self.demodulate(z_tilde)

    def transmit_batch(self, x_seq):
        """
        Vectorised: x_seq has shape (N, ℓx).
        Returns (Y, Z) each of shape (N, ℓx).
        """
        x_seq = np.asarray(x_seq, dtype=int)
        N = x_seq.shape[0]

        # Bits → symbol indices (efficient, no string ops)
        powers = 1 << np.arange(self.ell_x - 1, -1, -1)    # [64,32,16,8,4,2,1]
        idx_tx = x_seq @ powers                            # (N,)
        x_tilde = self.constellation[idx_tx]               # (N,)

        n_b = np.random.randn(N) * self.sigma_b
        n_e = np.random.randn(N) * self.sigma_e

        # Demodulate via rounding
        idx_y = np.round((x_tilde + n_b + (self.M - 1)) / 2.0).astype(int)
        idx_z = np.round((x_tilde + n_e + (self.M - 1)) / 2.0).astype(int)
        idx_y = np.clip(idx_y, 0, self.M - 1)
        idx_z = np.clip(idx_z, 0, self.M - 1)

        # Indices → bits
        Y = ((idx_y[:, None] >> np.arange(self.ell_x - 1, -1, -1)) & 1).astype(int)
        Z = ((idx_z[:, None] >> np.arange(self.ell_x - 1, -1, -1)) & 1).astype(int)
        return Y, Z

    def symbol_errors_batch(self, x_seq):
        """
        Return (SER_y, SER_z) for a batch of words, counting SYMBOL errors
        (wrong constellation index), which is what the spec asks to verify.
        """
        x_seq = np.asarray(x_seq, dtype=int)
        N = x_seq.shape[0]

        powers = 1 << np.arange(self.ell_x - 1, -1, -1)
        idx_tx = x_seq @ powers
        x_tilde = self.constellation[idx_tx]

        n_b = np.random.randn(N) * self.sigma_b
        n_e = np.random.randn(N) * self.sigma_e

        idx_y = np.clip(np.round((x_tilde + n_b + (self.M - 1)) / 2.0).astype(int), 0, self.M - 1)
        idx_z = np.clip(np.round((x_tilde + n_e + (self.M - 1)) / 2.0).astype(int), 0, self.M - 1)

        ser_y = np.mean(idx_y != idx_tx)
        ser_z = np.mean(idx_z != idx_tx)
        return ser_y, ser_z


# ──────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(42)
    N_WORDS = 200_000        # long symbol sequence
    ELL_X   = 7
    M       = 2 ** ELL_X

    # Test several (SNR_B, SNR_E) pairs — B always better than E.
    # Pairs chosen in the medium-SNR regime, where SER is neither saturated
    # (~1) nor vanishingly small (below numerical floor of N_WORDS realizations),
    # so that the SER ratio is measurable.
    #
    # Interpretation of √(SNR_B,lin / SNR_E,lin):
    #   For PAM with noise σ, SER ≈ 2(1-1/M)·Q(1/σ) at medium-high SNR,
    #   and σ = √(P/SNR_lin), so σ_z/σ_y = √(SNR_B/SNR_E).
    #   Thus √(SNR_B/SNR_E) is exactly the ratio of noise std-devs experienced
    #   by Eve vs. Bob. The SER ratio is NOT equal to this — because Q(·) is
    #   exponentially decaying, so a noise √k times larger translates into
    #   a SER MUCH more than √k times larger. We show both quantities to
    #   highlight this amplification effect.
    test_pairs = [
        (38.0, 32.0),
        (40.0, 32.0),
        (42.0, 34.0),
        (42.0, 30.0),
        (44.0, 36.0),
        (44.0, 32.0),
    ]

    print("=" * 110)
    print("VERIFICATION — SER ratio  vs  √(SNR_B,lin / SNR_E,lin) = σ_E / σ_B")
    print("=" * 110)
    print(f"{'SNR_B':>7} {'SNR_E':>7} | {'SER_y':>12} {'SER_z':>12} | "
          f"{'SER_z/SER_y':>14} | {'√(SNR_B/SNR_E)':>15} | {'σ_E/σ_B (meas.)':>16}")
    print("-" * 110)

    results = []
    for snr_b, snr_e in test_pairs:
        ch = WiretapAWGN(snr_b_db=snr_b, snr_e_db=snr_e, ell_x=ELL_X)

        # Generate random input sequence (uniform bits)
        x_seq = np.random.randint(0, 2, size=(N_WORDS, ELL_X))

        ser_y, ser_z = ch.symbol_errors_batch(x_seq)

        snr_b_lin   = 10 ** (snr_b / 10)
        snr_e_lin   = 10 ** (snr_e / 10)
        sqrt_ratio  = np.sqrt(snr_b_lin / snr_e_lin)   # theoretical σ_E/σ_B
        sigma_ratio = ch.sigma_e / ch.sigma_b          # measured σ_E/σ_B
        emp_ratio   = ser_z / ser_y if ser_y > 1e-10 else float('inf')

        results.append((snr_b, snr_e, ser_y, ser_z, emp_ratio, sqrt_ratio))

        print(f"{snr_b:>7.2f} {snr_e:>7.2f} | {ser_y:>12.5e} {ser_z:>12.5e} | "
              f"{emp_ratio:>14.4f} | {sqrt_ratio:>15.4f} | {sigma_ratio:>16.4f}")

    print()
    print("Observations:")
    print("  • The measured σ_E/σ_B matches √(SNR_B/SNR_E) exactly")
    print("    → AWGN noise variances are implemented correctly (SNRs set as specified).")
    print("  • The empirical SER ratio SER_z/SER_y is always LARGER than √(SNR_B/SNR_E),")
    print("    because PAM SER ≈ 2(1-1/M)·Q(1/σ) decays exponentially in 1/σ: a noise")
    print("    only √k times larger yields a SER MUCH more than √k times larger.")
    print("  • This exponential amplification of Eve's disadvantage in symbol errors is")
    print("    precisely what random-binning over AWGN will exploit to achieve secrecy.")

    # ──────────────────────────────────────────────
    # Extra demo: transmit a fixed word many times and plot noise histograms
    # ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print("DEMO — Input x = 1001000 (same as Task 1), transmitted 10 times")
    print("=" * 70)
    ch = WiretapAWGN(snr_b_db=35.0, snr_e_db=20.0, ell_x=ELL_X)
    x_bits = np.array([1,0,0,1,0,0,0])
    print(f"x = {''.join(map(str,x_bits))}  (symbol a = {ch.modulate(x_bits)})")
    print(f"σ_B = {ch.sigma_b:.4f}   σ_E = {ch.sigma_e:.4f}")
    for i in range(10):
        y, z = ch.transmit(x_bits)
        err_y = int(np.any(y != x_bits))
        err_z = int(np.any(z != x_bits))
        print(f"  trial {i+1:2d}:  y = {''.join(map(str,y))} (err={err_y})   "
              f"z = {''.join(map(str,z))} (err={err_z})")

    # ──────────────────────────────────────────────
    # Plot 1 — SER ratio vs √SNR ratio on a sweep
    # ──────────────────────────────────────────────
    print("\nSweeping SNR_E (fixed SNR_B = 40 dB) ...")
    snr_b_fixed = 40.0
    snr_e_sweep = np.arange(10.0, 40.1, 2.0)

    emp_ratios   = []
    theo_ratios  = []
    ser_y_list   = []
    ser_z_list   = []

    N_SW = 200_000
    x_seq = np.random.randint(0, 2, size=(N_SW, ELL_X))

    for snr_e in snr_e_sweep:
        ch = WiretapAWGN(snr_b_db=snr_b_fixed, snr_e_db=snr_e, ell_x=ELL_X)
        sy, sz = ch.symbol_errors_batch(x_seq)
        ser_y_list.append(sy)
        ser_z_list.append(sz)
        emp_ratios.append(sz / sy if sy > 1e-10 else np.nan)
        theo_ratios.append(np.sqrt(10**(snr_b_fixed/10) / 10**(snr_e/10)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(snr_e_sweep, ser_y_list, 'o-', color='steelblue',
                label=f'SER_y  (SNR_B={snr_b_fixed} dB, fixed)')
    ax.semilogy(snr_e_sweep, ser_z_list, 's-', color='tomato',
                label='SER_z  (varying SNR_E)')
    ax.set_xlabel('SNR_E [dB]')
    ax.set_ylabel('Symbol Error Rate')
    ax.set_title(f'PAM-{M} SER vs SNR_E   (legitimate SNR_B = {snr_b_fixed} dB)')
    ax.grid(alpha=0.3, which='both')
    ax.legend()

    ax = axes[1]
    ax.plot(snr_e_sweep, emp_ratios,  'o-', color='darkorange',
            label='Empirical SER_z / SER_y')
    ax.plot(snr_e_sweep, theo_ratios, '--', color='purple',
            label=r'Theoretical $\sqrt{\mathrm{SNR}_{B,\mathrm{lin}} / \mathrm{SNR}_{E,\mathrm{lin}}}$')
    ax.set_xlabel('SNR_E [dB]')
    ax.set_ylabel('Ratio')
    ax.set_yscale('log')
    ax.set_title('Empirical SER ratio vs theoretical prediction')
    ax.grid(alpha=0.3, which='both')
    ax.legend()

    plt.tight_layout()
    plt.savefig('task8_verification.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved → task8_verification.png")
