
""" Generate example codeword using generated code spec """

import numpy as np

OUT_PREFIX = "artifacts/168_83"

def build_H(CV6idx, CV7idx, n_vars):
    checks = list(CV6idx) + list(CV7idx)
    m = len(checks)
    H = np.zeros((m, n_vars), dtype=np.uint8)
    for i, c in enumerate(checks):
        H[i, c] = 1
    return H

def random_codeword(H):
    # Solve H c = 0 by Gaussian elimination (over GF(2))
    H2 = H.copy()
    m, n = H2.shape

    pivots = []
    row = 0
    for col in range(n):
        for r in range(row, m):
            if H2[r, col]:
                H2[[row, r]] = H2[[r, row]]
                pivots.append(col)
                for rr in range(m):
                    if rr != row and H2[rr, col]:
                        H2[rr] ^= H2[row]
                row += 1
                break
        if row == m:
            break

    free_vars = [i for i in range(n) if i not in pivots]
    c = np.zeros(n, dtype=np.uint8)
    c[free_vars] = np.random.randint(0, 2, size=len(free_vars))

    # back-substitute
    for r, col in reversed(list(enumerate(pivots))):
        s = np.dot(H2[r, col+1:], c[col+1:]) & 1
        c[col] = s

    return c

# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    CV6idx = np.load("artifacts/168_83_CV6idx.npy")
    CV7idx = np.load("artifacts/168_83_CV7idx.npy")

    H = build_H(CV6idx, CV7idx, n_vars=168)
    print("Generated H")
    
    cw = random_codeword(H)
    np.save(f"{OUT_PREFIX}_cw_example", cw)
    print("Generated random codeword")
