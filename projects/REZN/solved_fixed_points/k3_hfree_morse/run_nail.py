"""Standalone, restartable nail runner: nails G=9,13,17 with self-continuation,
saving P_morse_G{G}.npy and a checkpoint after EACH grid (so a kill/restart
resumes from the last completed grid).  Writes run.log."""
import os, sys, json, time
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import morse_solver as S
from morse_solver import UMAX, SymReducer3

LOG = open(os.path.join(HERE, "run.log"), "a")
def log(*a):
    m = " ".join(str(x) for x in a); print(m, flush=True); LOG.write(m + "\n"); LOG.flush()


def interp_up(Pprev, Gprev, Gnew):
    axp = np.linspace(-UMAX, UMAX, Gprev); axn = np.linspace(-UMAX, UMAX, Gnew)
    rgi = RegularGridInterpolator((axp,) * 3, Pprev, bounds_error=False, fill_value=None)
    A, B, C = np.meshgrid(axn, axn, axn, indexing="ij")
    return np.clip(rgi(np.column_stack([A.ravel(), B.ravel(), C.ravel()])).reshape((Gnew,) * 3),
                   1e-9, 1 - 1e-9)


def main():
    TAU, GAMMA = 2.0, 0.1
    G_LIST = [9, 13, 17]
    log("=" * 70)
    log(f"MORSE NAIL RUNNER tau={TAU} gamma={GAMMA} eps_c={S.EPS_C} G_LIST={G_LIST}  "
        f"{time.strftime('%H:%M:%S')}")
    rows = []
    ckpt = os.path.join(HERE, "_nail_ckpt.json")
    if os.path.exists(ckpt):
        rows = json.load(open(ckpt)).get("rows", [])
    done_G = {r["G"] for r in rows}
    Pprev, Gprev = None, None
    # reload last completed P for continuation
    for r in rows:
        p = os.path.join(HERE, f"P_morse_G{r['G']}.npy")
        if os.path.exists(p):
            Pprev, Gprev = np.load(p), r["G"]

    for G in G_LIST:
        if G in done_G and any(r["G"] == G and r["Finf"] < 1e-9 for r in rows):
            log(f"  G={G}: already nailed (resume) -> skip")
            p = os.path.join(HERE, f"P_morse_G{G}.npy")
            if os.path.exists(p):
                Pprev, Gprev = np.load(p), G
            continue
        t = time.time()
        x0 = None
        if Pprev is not None:
            red = SymReducer3(G)
            x0 = red.reduce(interp_up(Pprev, Gprev, G))
        last = {"f": 1e9}

        def vb(n, f, G=G, last=last):
            if f < last["f"] * 0.5 or f < 1e-6 or n % 20 == 0:
                log(f"    G={G} it {n} ||F||={f:.3e}")
                last["f"] = f
        P, sol, Finf, info = S.solve_morse_pr(
            G, TAU, GAMMA, x0_red=x0, f_tol=1e-9, verbose=vb)
        info["walltime_s"] = round(time.time() - t, 1)
        np.save(os.path.join(HERE, f"P_morse_G{G}.npy"), P)
        Pprev, Gprev = P, G
        rows = [r for r in rows if r["G"] != G] + [info]
        rows.sort(key=lambda r: r["G"])
        json.dump({"rows": rows}, open(ckpt, "w"), indent=2)
        log(f"  G={G:2d}: ||F||={Finf:.3e} {'NAILED' if Finf < 1e-9 else 'FLOOR'} "
            f"deficit={info['deficit']:.5f} d_FR={info['d_FR']:.4f} "
            f"iters={info['iters']} picard={info['used_picard']} ({info['walltime_s']}s)")
    log("NAILRUN DONE  " + "  ".join(f"G{r['G']}:{r['Finf']:.2e}" for r in rows))


if __name__ == "__main__":
    main()
