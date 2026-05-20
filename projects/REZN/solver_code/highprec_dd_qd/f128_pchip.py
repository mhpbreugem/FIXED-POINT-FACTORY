"""float128 (80-bit long double) PCHIP, matching scipy.PchipInterpolator.
Vectorized __call__ for array queries (the speed win over scalar mpmath)."""
import numpy as np
F = np.float128


def _pchip_slopes(x, y):
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n, dtype=F)
    for k in range(1, n-1):
        if delta[k-1] == 0 or delta[k] == 0 or (delta[k-1] > 0) != (delta[k] > 0):
            d[k] = F(0)
        else:
            w1 = 2*h[k] + h[k-1]; w2 = h[k] + 2*h[k-1]
            d[k] = (w1+w2) / (w1/delta[k-1] + w2/delta[k])
    def edge(h0, h1, m0, m1):
        dd = ((2*h0+h1)*m0 - h0*m1)/(h0+h1)
        if (dd > 0) != (m0 > 0): return F(0)
        if ((m0 > 0) != (m1 > 0)) and (abs(dd) > 3*abs(m0)): return 3*m0
        return dd
    if n > 2:
        d[0] = edge(h[0], h[1], delta[0], delta[1])
        d[-1] = edge(h[-1], h[-2], delta[-1], delta[-2])
    else:
        d[0] = delta[0]; d[-1] = delta[0]
    return h, delta, d


class F128Pchip:
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=F)
        self.y = np.asarray(y, dtype=F)
        self.h, self.delta, self.d = _pchip_slopes(self.x, self.y)
        self.n = len(self.x)

    def __call__(self, xq):
        scalar = np.isscalar(xq) or (getattr(xq, "ndim", 1) == 0)
        xq = np.atleast_1d(np.asarray(xq, dtype=F))
        x = self.x; n = self.n
        k = np.searchsorted(x, xq) - 1
        k = np.clip(k, 0, n-2)
        hk = self.h[k]
        t = (xq - x[k]) / hk
        h00 = (1+2*t)*(1-t)**2
        h10 = t*(1-t)**2
        h01 = t*t*(3-2*t)
        h11 = t*t*(t-1)
        out = (h00*self.y[k] + h10*hk*self.d[k]
               + h01*self.y[k+1] + h11*hk*self.d[k+1])
        return out[0] if scalar else out
