# SPIN — Spectral Preconditioning via IN-span learning

**Reduced-order models can learn from their own predictions.**

This repository introduces **in-span learning**, a new way for a computational
model to adapt online, and a reduced-order model (ROM) that uses it, called
**SPIN**.

> 📄 **Paper:** *In-span learning: adapting reduced-order models using their own
> predictions*, A. Hedayat, L. Balzano, K. Duraisamy.
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) *(link will be updated on posting)*

---

## The idea in one minute

A reduced-order model compresses an expensive high-dimensional simulation into a
small subspace and evolves only a few coordinates inside it. It is fast, but it
loses accuracy once the dynamics drift beyond the data it was trained on.

**Adaptive ROMs** fix this by updating their subspace online using *external*
information — an occasional full-order solve, a sensor snapshot, a correction.
Such information lives **outside** the current subspace, so we call it
**out-of-span**. This is the standard, state-of-the-art recipe (our *baseline
adaptive ROM*).

The key observation of this work: between those external corrections, the ROM is
constantly producing predictions of its own. By construction these predictions
live **inside** the current subspace — they are **in-span** — so the usual
subspace-update view says they carry no new information and discards them.

We show that is wrong. If you stream the ROM's **own in-span predictions** through
an incremental SVD with forgetting, you don't move the subspace (you can't — the
predictions are already in it), but you **rotate and reweight the basis inside
it**. This is a *trajectory-informed spectral preconditioner*: it changes how the
basis is prepared to absorb the **next** external correction, and it makes that
correction land far more effectively.

```
  baseline adaptive ROM:   predict ─ predict ─ predict ─ CORRECT ─ predict ─ ...
                                                          ▲ external (out-of-span)

  SPIN ROM:                predict ─ in-span ─ in-span ─ CORRECT ─ in-span ─ ...
                                     ▲ learn from   ▲ same external correction,
                                       own output     but absorbed better
```

**SPIN is exactly the baseline adaptive ROM, plus an in-span update between every
pair of external corrections.** Same correction budget, no extra full-order
solves — just more use of information the model already generated.

> SPIN's out-of-span channel (the baseline adaptive ROM we compare against) is
> itself a strong, recent method — see our previous work,
> [arXiv:2605.28684](https://arxiv.org/abs/2605.28684). SPIN adds the in-span
> channel on top of it.

---

## What's in here

```
SPIN/
├── src/spin/              # the library — all the machinery
│   ├── models.py          #   full-order models: Spiral, Burgers, Fisher-KPP
│   ├── isvd.py            #   incremental SVD with forgetting (the core update)
│   ├── linalg.py          #   POD basis + QDEIM hyper-reduction
│   ├── rom.py             #   the adaptive ROM driver: static / baseline / SPIN
│   ├── diagnostics.py     #   the metrics used in the paper's figures
│   ├── spiral.py          #   the closed-form spiral experiment
│   └── plotting.py        #   shared figure style + helpers
└── examples/              # three self-contained, runnable notebooks
    ├── 01_spiral.ipynb       #   the toy example that exposes the mechanism
    ├── 02_burgers.ipynb      #   viscous Burgers, long-horizon prediction
    └── 03_fisher_kpp.ipynb   #   Fisher-KPP fronts, long-horizon prediction
```

The three notebooks reproduce the main-text results of the paper (the spiral
mechanism, the Burgers and Fisher-KPP predictions, and the spectral diagnostics).

---

## Install

```bash
git clone https://github.com/USERNAME/SPIN.git
cd SPIN
pip install -e .            # installs numpy, scipy, matplotlib
# for the notebooks:
pip install -e ".[notebooks]"
```

Requires Python ≥ 3.9.

---

## Quickstart

Build a ROM and run it in any of the three modes — `static`, `baseline`, or
`spin` — from the *same* driver:

```python
import numpy as np
from spin.models import BurgersSolver
from spin.linalg import compute_pod_basis, qdeim
from spin.rom import BurgersROM
from spin.diagnostics import relative_l2_error

# 1. full-order model + a short training window
solver = BurgersSolver(Nx=256, nu=1e-2, dt=1e-3)
snaps  = np.stack(solver.simulate(solver.initial_condition(), n_steps=500,
                                  tol=1e-8), axis=1)

# 2. offline rank-4 POD basis + QDEIM samples from only the first 4 snapshots
Phi0, sigma0 = compute_pod_basis(snaps[:, :4], r=4)
p_inds       = qdeim(Phi0, n_sensors=4)
a0           = Phi0.T @ snaps[:, 0]

# 3. run baseline adaptive vs SPIN (same correction interval zs=10)
def run(mode, gamma_in, gamma_out):
    rom = BurgersROM(Phi0.copy(), sigma0.copy(), p_inds.copy(), solver,
                     dt=1e-3, zs=10, mode=mode,
                     gamma_in=gamma_in, gamma_out=gamma_out, nu=1e-2)
    return rom.simulate(a0.copy(), n_steps=500)

baseline = run("baseline", gamma_in=1.0, gamma_out=0.01)
spin     = run("spin",     gamma_in=1.0, gamma_out=0.25)

print("baseline adaptive:", relative_l2_error(baseline, snaps).mean())  # ~5.5e-2
print("SPIN             :", relative_l2_error(spin,     snaps).mean())  # ~1.9e-2
```

The only thing that changes between the two runs is `mode` (and the forgetting
factors): `"baseline"` uses out-of-span corrections only, `"spin"` adds the
in-span channel.

### The three modes

| `mode` | in-span updates | out-of-span corrections | what it is |
|---|:---:|:---:|---|
| `"static"`   | ✗ | ✗ | fixed POD basis, no adaptation |
| `"baseline"` | ✗ | ✓ | baseline adaptive ROM (state-of-the-art) |
| `"spin"`     | ✓ | ✓ | **SPIN** (this work) |

### Key knobs

- `zs` — how often an external (out-of-span) correction is taken (every `zs` steps).
- `gamma_out` — forgetting for the out-of-span update (how much past history the
  correction discounts).
- `gamma_in` — forgetting for the in-span update (the memory horizon of in-span
  learning). `gamma_in ≈ 1` reinforces several modes; small `gamma_in` aggressively
  suppresses inactive ones.

---

## Reproducing the paper

Open the notebooks in `examples/` and run top to bottom. Each one has all of its
control variables in a single cell near the top, so you can immediately change
the rank, correction interval, or forgetting factors and see the effect. With the
default values they reproduce the main-text numbers:

| experiment | static | baseline adaptive | SPIN |
|---|---|---|---|
| **Burgers** (mean rel. $L_2$) | 3.7e-1 | 5.5e-2 | **1.9e-2** |
| **Fisher-KPP** (mean rel. $L_2$) | 2.1e-1 | 1.2e-1 | **7.8e-3** |

For the spiral, the notebook reproduces the reported residual capture
(0.26 → 0.80), plane-change angle (15° → 53°), and correction error
(1.0e-2 → 8.8e-4) when in-span preconditioning is enabled.

---

## Citing

If you use this code, please cite the paper:

```bibtex
@article{hedayat2026inspan,
  title   = {In-span learning: adapting reduced-order models using their own predictions},
  author  = {Hedayat, Amirpasha and Balzano, Laura and Duraisamy, Karthik},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
