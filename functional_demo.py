""" Load the generated random codeword and check vars lists,
    simulate transmission and reception over a BPSK AWGN channel -> LLRs, 
    and decode using PyLDPC decoder

    Notes:
    - LLR convention: positive LLR indicates bit = 1
    - BPSK mapping used here is: 0 -> -1, 1 -> +1
    
"""

import numpy as np

def awgn_llr(codeword, EbN0_dB):
    x = 2.0 * codeword - 1.0
    EbN0 = 10 ** (EbN0_dB / 10)
    sigma = np.sqrt(1 / (2 * EbN0))
    y = x + sigma * np.random.randn(len(x))
    llr = 2 * y / (sigma ** 2)
    return llr.astype(np.float32)


from PyLDPC_bp_decoder import LdpcDecoder

CV6idx = np.load("artifacts/168_83_CV6idx.npy")
CV7idx = np.load("artifacts/168_83_CV7idx.npy")

dec = LdpcDecoder(CV6idx, CV7idx)

cw = np.load("artifacts/168_83_cw_example.npy")
llr = awgn_llr(cw, EbN0_dB = 1)

for i in range(12):
    llr, ncheck = dec.do_ldpc_iteration(llr)
    print(f"iter {i}: ncheck={ncheck}")
    if ncheck == 0:
        break

