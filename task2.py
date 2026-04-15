import numpy as np

HAMMING_CODE = [
    "0000000", "1000110", "0100101", "1100011",
    "0010011", "1010101", "0110110", "1110000",
    "0001111", "1001001", "0101010", "1101100",
    "0011100", "1011010", "0111001", "1111111"
]

class RandomBinningEncoder:
    def __init__(self, codebook):
        self.codebook = codebook
        self.bins = self._build_bins()

    def _build_bins(self):
        bins = {}
        for d_int in range(8):
            d_str    = format(d_int, '03b')
            d_bar    = format(d_int ^ 0b111, '03b')
            prefix_0 = '0' + d_str
            prefix_1 = '1' + d_bar
            bins[d_str] = [cw for cw in self.codebook
                           if cw[:4] == prefix_0 or cw[:4] == prefix_1]
        return bins

    def encode(self, d_str):
        bin_cw = self.bins[d_str]
        return bin_cw[np.random.randint(len(bin_cw))]

    def print_bins(self):
        print("Bins (message -> codewords):")
        print(f"  {'d':>5}  {'prefix_0':>8}  {'prefix_1':>8}  {'codewords'}")
        print("  " + "-"*55)
        for d_int in range(8):
            d_str = format(d_int, '03b')
            d_bar = format(d_int ^ 0b111, '03b')
            p0    = '0' + d_str
            p1    = '1' + d_bar
            cws   = self.bins[d_str]
            print(f"  {d_str:>5}  {p0:>8}  {p1:>8}  {cws}")

if __name__ == '__main__':
    encoder = RandomBinningEncoder(HAMMING_CODE)
    encoder.print_bins()

    print("\nBin size check:")
    all_ok = True
    for d_int in range(8):
        d_str = format(d_int, '03b')
        size  = len(encoder.bins[d_str])
        ok    = "✓" if size == 2 else "✗ ERROR"
        print(f"  d={d_str}  bin size={size}  {ok}")
        if size != 2:
            all_ok = False
    print(f"\n  All bins have size 2: {'YES ✓' if all_ok else 'NO ✗'}")

    print("\nEncoding each message 6 times (should alternate between the 2 codewords):")
    print(f"  {'d':>5}  {'bin':>25}  outputs")
    print("  " + "-"*60)
    for d_int in range(8):
        d_str   = format(d_int, '03b')
        cws     = encoder.bins[d_str]
        outputs = [encoder.encode(d_str) for _ in range(6)]
        print(f"  {d_str:>5}  {str(cws):>25}  {outputs}")