import sys
sys.path.append('.')

from task1 import WiretapUniformErrorChannel
from task2 import RandomBinningEncoder, HAMMING_CODE


# ──────────────────────────────────────────────
# Task 3 — Random Binning Decoder
# ──────────────────────────────────────────────

def hamming_distance(a, b):
    return sum(x != y for x, y in zip(a, b))

class RandomBinningDecoder:
    def __init__(self, codebook):
        self.codebook = codebook

    def decode(self, y_str):
        # Step 1: find closest codeword by minimum Hamming distance
        x_hat = min(self.codebook, key=lambda cw: hamming_distance(y_str, cw))

        # Step 2: recover message from first bit
        x1    = x_hat[0]
        bits  = x_hat[1:4]   # bits x̂2, x̂3, x̂4

        if x1 == '0':
            u_hat = bits
        else:
            u_hat = ''.join('1' if b == '0' else '0' for b in bits)  # ones' complement

        return u_hat, x_hat


if __name__ == '__main__':

    channel = WiretapUniformErrorChannel(n=7, r=1, s=3)
    encoder = RandomBinningEncoder(HAMMING_CODE)
    decoder = RandomBinningDecoder(HAMMING_CODE)

    # ──────────────────────────────────────────────
    # Demo 1: encoder → decoder (no channel), 3 volte per messaggio
    # ──────────────────────────────────────────────
    print("=" * 75)
    print("CHAIN 1: encoder → decoder (no channel), 3 trials per message")
    print("=" * 75)
    for d_int in range(8):
        d_str = format(d_int, '03b')
        print(f"\n  d = {d_str}  |  bin = {encoder.bins[d_str]}")
        for _ in range(3):
            x            = encoder.encode(d_str)
            u_hat, x_hat = decoder.decode(x)
            result       = 'OK' if u_hat == d_str else 'ERROR'
            print(f"    x={x}  x_hat={x_hat}  u_hat={u_hat}  [{result}]")

    # ──────────────────────────────────────────────
    # Demo 2: encoder → legit channel → decoder, 3 volte per messaggio
    # ──────────────────────────────────────────────
    print()
    print("=" * 75)
    print("CHAIN 2: encoder → legitimate channel (r=1) → decoder, 3 trials per message")
    print("=" * 75)
    for d_int in range(8):
        d_str = format(d_int, '03b')
        print(f"\n  d = {d_str}  |  bin = {encoder.bins[d_str]}")
        for _ in range(3):
            x          = encoder.encode(d_str)
            x_bits     = tuple(int(b) for b in x)
            y_bits, _  = channel.transmit(x_bits)
            y_str      = ''.join(str(b) for b in y_bits)
            u_hat, x_hat = decoder.decode(y_str)
            error_vec  = ''.join(str(a ^ b) for a, b in zip(x_bits, y_bits))
            n_errors   = sum(a ^ b for a, b in zip(x_bits, y_bits))
            result     = 'OK' if u_hat == d_str else 'ERROR'
            print(f"    x={x}  error={error_vec}({n_errors})  y={y_str}  x_hat={x_hat}  u_hat={u_hat}  [{result}]")

    # ──────────────────────────────────────────────
    # Verifica statistica
    # ──────────────────────────────────────────────
    N_TRIALS = 100

    print()
    print("=" * 75)
    print("STATISTICAL CHECK — chain 1: encoder → decoder")
    print("=" * 75)
    errors = 0
    for d_int in range(8):
        d_str = format(d_int, '03b')
        for _ in range(N_TRIALS):
            x            = encoder.encode(d_str)
            u_hat, x_hat = decoder.decode(x)
            if u_hat != d_str:
                errors += 1
                print(f"  ERROR: d={d_str}  x={x}  x_hat={x_hat}  u_hat={u_hat}")
    print(f"Total errors: {errors} / {8 * N_TRIALS}  ({'PASS' if errors == 0 else 'FAIL'})")

    print()
    print("=" * 75)
    print("STATISTICAL CHECK — chain 2: encoder → legitimate channel → decoder")
    print("=" * 75)
    errors = 0
    for d_int in range(8):
        d_str = format(d_int, '03b')
        for _ in range(N_TRIALS):
            x             = encoder.encode(d_str)
            x_bits        = tuple(int(b) for b in x)
            y_bits, _     = channel.transmit(x_bits)
            y_str         = ''.join(str(b) for b in y_bits)
            u_hat, x_hat  = decoder.decode(y_str)
            if u_hat != d_str:
                errors += 1
                print(f"  ERROR: d={d_str}  x={x}  y={y_str}  x_hat={x_hat}  u_hat={u_hat}")
    print(f"Total errors: {errors} / {8 * N_TRIALS}  ({'PASS' if errors == 0 else 'FAIL'})")