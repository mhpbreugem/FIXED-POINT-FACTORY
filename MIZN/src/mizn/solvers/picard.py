"""Damped Picard iteration:  P <- (1-omega)*P + omega*P_new."""
import numpy as np


def update_picard(P, P_new, params):
    omega = params.omega
    return (1.0 - omega) * P + omega * P_new
