"""Inspect what makes NQ=16 special at G=7.
Look at GL nodes vs Lobatto nodes - are there near-coincidences?"""
import sys
sys.path.insert(0, '/tmp/cheby_h0')
import numpy as np
from cheby_numba import LOBATTO

print(f'Lobatto nodes at G=7 (xi_lobatto = -cos(pi*j/6)):')
print(LOBATTO)

print('\nGL nodes at various NQ:')
for NQ in [12, 13, 14, 15, 16, 17, 18, 20, 24]:
    gl, _ = np.polynomial.legendre.leggauss(NQ)
    # check for near-coincidence with Lobatto nodes
    min_gap = float('inf')
    for lob in LOBATTO:
        gap = float(np.min(np.abs(gl - lob)))
        if gap < min_gap: min_gap = gap
    # also: minimum gap between any GL node and any other GL node
    print(f'  NQ={NQ:2d}: min |GL - Lobatto| = {min_gap:.4f}')

# Closer look at NQ=16
print('\nFor NQ=16, gaps to each Lobatto node:')
gl, _ = np.polynomial.legendre.leggauss(16)
for lob in LOBATTO:
    dists = np.abs(gl - lob)
    closest = np.argmin(dists)
    print(f'  Lobatto={lob:+.4f}: closest GL node = {gl[closest]:+.4f}, '
          f'gap = {dists[closest]:.4f}')

print('\nFor NQ=12, gaps to each Lobatto node (BAD):')
gl, _ = np.polynomial.legendre.leggauss(12)
for lob in LOBATTO:
    dists = np.abs(gl - lob)
    closest = np.argmin(dists)
    print(f'  Lobatto={lob:+.4f}: closest GL node = {gl[closest]:+.4f}, '
          f'gap = {dists[closest]:.4f}')
