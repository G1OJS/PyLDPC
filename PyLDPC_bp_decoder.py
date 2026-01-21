import numpy as np

class LdpcDecoder:
    def __init__(self, CV6idx, CV7idx):
        self.CV6idx =   CV6idx
        self.CV7idx =   CV7idx
        self.mC2V_prev6 = None
        self.mC2V_prev7 = None
        
    def calc_ncheck(self, llr):
        bits6 = llr[self.CV6idx] > 0
        self.parity6 = np.sum(bits6, axis=1) & 1
        bits7 = llr[self.CV7idx] > 0
        self.parity7 = np.sum(bits7, axis=1) & 1
        return int(np.sum(self.parity7) + np.sum(self.parity6))

    def _pass_messages(self, llr, CVidx, mC2V_prev, update_collector):
        if mC2V_prev is None:
            mC2V_prev = np.zeros(CVidx.shape, dtype=np.float32)
        mV2C = llr[CVidx] - mC2V_prev
        tanh_mV2C = np.tanh(-mV2C)
        tanh_mC2V = np.prod(tanh_mV2C, axis=1, keepdims=True) / tanh_mV2C
        alpha_atanh_approx = 1.18
        mC2V_curr  = tanh_mC2V / ((tanh_mC2V - alpha_atanh_approx) * (alpha_atanh_approx + tanh_mC2V))
        np.add.at(update_collector, CVidx, mC2V_curr - mC2V_prev)
        return mC2V_curr
    
    def do_ldpc_iteration(self, llr):
        update_collector = np.zeros_like(llr)
        self.mC2V_prev6 = self._pass_messages(llr, self.CV6idx, self.mC2V_prev6, update_collector)
        self.mC2V_prev7 = self._pass_messages(llr, self.CV7idx, self.mC2V_prev7, update_collector)
        llr += update_collector
        return llr, self.calc_ncheck(llr)
