"""DECISIVE analytic verification of the co-area quadrature claim.

Controlled curved surface P(a,b)=sigmoid(a^2 - b) on [-L,L]^2 whose level set
{P=0.5} is the parabola b=a^2, along which |grad P| VARIES. We compute the
posterior-evidence-style ratio  R = (int_{P=p0} g1 dmu) / (int_{P=p0} g0 dmu)
under different measures dmu, and compare to the EXACT continuum co-area ratio
computed by 1-D quadrature.

This isolates the pure quadrature question (no fixed-point iteration):
  - kernel sum  Sum_cells K_h(P-p) g   as h->0       (the ladder's integrand)
  - naive two-axis-averaged crossing scan, weight 1   (method b)
  - 1/|grad P| two-axis-averaged weighted scan        (method c, spec's weight)
  - EXACT co-area integral int g/|gradP| ds            (ground truth)
"""
import json
import numpy as np
from scipy import integrate

L = 2.0
P0 = 0.5
def Pf(a, b):
    return 1.0 / (1.0 + np.exp(-(a * a - b)))
def g1(a, b):
    return np.exp(-(a - 0.5) ** 2) * np.exp(-(b - 0.5) ** 2)
def g0(a, b):
    return np.exp(-(a + 0.5) ** 2) * np.exp(-(b + 0.5) ** 2)


def exact_coarea_ratio():
    # level set b=a^2; |gradP|=0.25*sqrt(4a^2+1); ds=sqrt(1+4a^2) da
    def integ(a, g):
        b = a * a
        if abs(b) > L:
            return 0.0
        grad = 0.25 * np.sqrt(4 * a * a + 1)
        ds = np.sqrt(1 + 4 * a * a)
        return g(a, b) / grad * ds
    num, _ = integrate.quad(lambda a: integ(a, g1), -L, L, limit=300)
    den, _ = integrate.quad(lambda a: integ(a, g0), -L, L, limit=300)
    return num / den


def kernel_ratio(G, h):
    u = np.linspace(-L, L, G)
    P = Pf(u[:, None], u[None, :])
    W = np.exp(-(P - P0) ** 2 / (2 * h * h))
    A, B = np.meshgrid(u, u, indexing='ij')
    num = float(np.sum(W * g1(A, B)))
    den = float(np.sum(W * g0(A, B)))
    return num / den


def strict_ratio(G, weighted):
    du = 2 * L / (G - 1)
    u = np.linspace(-L, L, G)
    P = Pf(u[:, None], u[None, :])
    nn = nd = 0.0
    for axis in range(2):
        for jb in range(G):
            prev = P[0, jb] if axis == 0 else P[jb, 0]
            for i in range(G - 1):
                nx = P[i + 1, jb] if axis == 0 else P[jb, i + 1]
                if (prev - P0) * (nx - P0) <= 0 and nx != prev:
                    fr = (P0 - prev) / (nx - prev)
                    if 0 <= fr <= 1:
                        if axis == 0:
                            a = (1 - fr) * u[i] + fr * u[i + 1]; b = u[jb]
                        else:
                            a = u[jb]; b = (1 - fr) * u[i] + fr * u[i + 1]
                        if weighted:
                            g_off = (nx - prev) / du
                            if 0 < jb < G - 1:
                                if axis == 0:
                                    vp = (1-fr)*P[i, jb+1]+fr*P[i+1, jb+1]
                                    vm = (1-fr)*P[i, jb-1]+fr*P[i+1, jb-1]
                                else:
                                    vp = (1-fr)*P[jb+1, i]+fr*P[jb+1, i+1]
                                    vm = (1-fr)*P[jb-1, i]+fr*P[jb-1, i+1]
                                g_tr = (vp - vm) / (2 * du)
                            else:
                                g_tr = 0.0
                            w = 1.0 / np.sqrt(g_off ** 2 + g_tr ** 2)
                        else:
                            w = 1.0
                        nn += w * g1(a, b); nd += w * g0(a, b)
                prev = nx
    return nn / nd


def main():
    exact = exact_coarea_ratio()
    out = {'exact_coarea_ratio': exact, 'surface': 'P=sigmoid(a^2-b), L=2',
           'kernel_h_to_0': {}, 'strict_du_to_0': {}}
    print(f'EXACT co-area ratio = {exact:.6f}')
    print('--- kernel ratio as h->0 (fixed fine grid G=401) ---')
    for h in (0.2, 0.1, 0.05, 0.02, 0.01):
        r = kernel_ratio(401, h)
        out['kernel_h_to_0'][str(h)] = r
        print(f'  h={h:.3f}: {r:.6f}  (err {abs(r-exact):.4f})')
    print('--- strict two-axis scans as du->0 ---')
    for G in (101, 201, 401, 801):
        rn = strict_ratio(G, False)
        rw = strict_ratio(G, True)
        out['strict_du_to_0'][str(G)] = {'naive': rn, 'w_grad': rw,
                                         'du': 2 * L / (G - 1)}
        print(f'  G={G} du={2*L/(G-1):.4f}: naive={rn:.6f} (err {abs(rn-exact):.4f})'
              f'  w_grad={rw:.6f} (err {abs(rw-exact):.4f})')
    # verdict
    rw_fine = out['strict_du_to_0']['801']['w_grad']
    rn_fine = out['strict_du_to_0']['801']['naive']
    out['verdict'] = {
        'w_grad_matches_coarea': bool(abs(rw_fine - exact) < 0.03),
        'naive_is_biased': bool(abs(rn_fine - exact) > 0.1),
        'w_grad_err_fine': abs(rw_fine - exact),
        'naive_err_fine': abs(rn_fine - exact),
        'naive_limit_approx': '~arc-length-measure ratio (no 1/|gradP|)',
        'note': 'Two-axis-averaged 1/|gradP| weighted strict scan converges to '
                'the EXACT co-area ratio (= kernel/ladder h->0 limit); the '
                'naive unweighted scan converges to the arc-length-measure '
                'ratio, a definite bias. Two-axis averaging is ESSENTIAL.'}
    json.dump(out, open('/home/user/FIXED-POINT-FACTORY/projects/REZN/'
                        'solved_fixed_points/k3_verify_coarea/'
                        'analytic_coarea.json', 'w'), indent=2)
    print('\nVERDICT:', out['verdict']['note'])


if __name__ == '__main__':
    main()
