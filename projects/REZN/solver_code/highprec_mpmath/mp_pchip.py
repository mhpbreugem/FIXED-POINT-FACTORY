"""mpmath PCHIP (Fritsch-Carlson monotone cubic Hermite) matching scipy.PchipInterpolator."""
import mpmath as mp


def _pchip_slopes(x, y):
    n = len(x)
    h = [x[k+1]-x[k] for k in range(n-1)]
    delta = [(y[k+1]-y[k])/h[k] for k in range(n-1)]
    d = [mp.mpf(0)]*n
    # interior
    for k in range(1, n-1):
        if delta[k-1] == 0 or delta[k] == 0 or (delta[k-1] > 0) != (delta[k] > 0):
            d[k] = mp.mpf(0)
        else:
            w1 = 2*h[k] + h[k-1]
            w2 = h[k] + 2*h[k-1]
            d[k] = (w1+w2) / (w1/delta[k-1] + w2/delta[k])
    # endpoints (scipy _edge_case)
    def edge(h0, h1, m0, m1):
        d_ = ((2*h0+h1)*m0 - h0*m1)/(h0+h1)
        if (d_ > 0) != (m0 > 0):
            return mp.mpf(0)
        if ((m0 > 0) != (m1 > 0)) and (abs(d_) > 3*abs(m0)):
            return 3*m0
        return d_
    if n > 2:
        d[0] = edge(h[0], h[1], delta[0], delta[1])
        d[-1] = edge(h[-1], h[-2], delta[-1], delta[-2])
    else:
        d[0] = delta[0]; d[-1] = delta[0]
    return h, delta, d


class MpPchip:
    def __init__(self, x, y):
        self.x = [mp.mpf(v) for v in x]
        self.y = [mp.mpf(v) for v in y]
        self.h, self.delta, self.d = _pchip_slopes(self.x, self.y)
        self.n = len(self.x)

    def __call__(self, xq):
        x = self.x; n = self.n
        # locate interval (clamp + linear cubic extrapolation like scipy extrapolate=True)
        if xq <= x[0]:
            k = 0
        elif xq >= x[-1]:
            k = n-2
        else:
            lo, hi = 0, n-1
            while hi - lo > 1:
                mid = (lo+hi)//2
                if x[mid] <= xq: lo = mid
                else: hi = mid
            k = lo
        # cubic Hermite on [x_k, x_{k+1}]
        hk = self.h[k]
        t = (xq - x[k])/hk
        h00 = (1+2*t)*(1-t)**2
        h10 = t*(1-t)**2
        h01 = t*t*(3-2*t)
        h11 = t*t*(t-1)
        return (h00*self.y[k] + h10*hk*self.d[k]
                + h01*self.y[k+1] + h11*hk*self.d[k+1])
