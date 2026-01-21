""" Load the generated random codeword and check vars lists,
    generate llr in the presence of added noise, 
    and decode using PyLDPC decoder
"""

import numpy as np

def awgn_llr(codeword, snr_db):
    x = 2.0 * codeword - 1.0
    snr = 10 ** (snr_db / 10)
    sigma = np.sqrt(1 / (2 * snr))
    y = x + sigma * np.random.randn(len(x))
    llr = 2 * y / (sigma ** 2)
    return llr.astype(np.float32)


from PyLDPC_bp_decoder import LdpcDecoder

CV6idx = np.load("artifacts/168_83_CV6idx.npy")
CV7idx = np.load("artifacts/168_83_CV7idx.npy")

dec = LdpcDecoder(CV6idx, CV7idx)

cw = np.load("artifacts/168_83_cw_example.npy")
llr = awgn_llr(cw, snr_db = 1)

for i in range(12):
    llr, ncheck = dec.do_ldpc_iteration(llr)
    print(f"iter {i}: ncheck={ncheck}")
    if ncheck == 0:
        break

