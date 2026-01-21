
""" Use code spec parameters to generate constants
needed for encoding and decoding """

import numpy as np
import json
import hashlib

# -----------------------
# Spec parameters
# -----------------------

N_VARS = 168

N_CHECKS_DEG6 = 59
N_CHECKS_DEG7 = 24

VAR_DEG_DISTR = {
    3: 150,
    4: 18,
}

SEED = 0x5A17C0DE

OUT_PREFIX = "artifacts/168_83"

# -----------------------
# Graph generation
# -----------------------

def generate_graph():
    rng = np.random.default_rng(SEED)

    # variable stubs
    var_stubs = []
    v = 0
    for deg, count in VAR_DEG_DISTR.items():
        for _ in range(count):
            var_stubs.extend([v] * deg)
            v += 1

    assert v == N_VARS

    rng.shuffle(var_stubs)

    # check stubs
    check_degs = ([6] * N_CHECKS_DEG6) + ([7] * N_CHECKS_DEG7)
    checks = []

    idx = 0
    for deg in check_degs:
        # retry until no duplicates inside the check
        for _ in range(1000):
            c = var_stubs[idx:idx + deg]
            if len(set(c)) == deg:
                checks.append(list(c))
                idx += deg
                break
            rng.shuffle(var_stubs)
        else:
            raise RuntimeError("Failed to construct check without duplicates")

    assert idx == len(var_stubs)
    return checks


# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    checks = generate_graph()

    checks6 = [c for c in checks if len(c) == 6]
    checks7 = [c for c in checks if len(c) == 7]

    CV6idx = np.array(checks6, dtype=np.int32)
    CV7idx = np.array(checks7, dtype=np.int32)

    np.save(f"{OUT_PREFIX}_CV6idx.npy", CV6idx)
    np.save(f"{OUT_PREFIX}_CV7idx.npy", CV7idx)

    print("Generated LDPC check lists")







    
