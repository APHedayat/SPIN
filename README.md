# SPIN — Spectral Preconditioning via IN-span learning

**SPIN** is a lightweight, modular Python framework for **online adaptive,
hyper-reduced reduced-order modeling (ROM)** of time-dependent nonlinear PDEs.
It introduces **in-span learning**: a reduced model can adapt not only from
external corrections, but also from the trajectory it has *already produced
itself*. Streaming the ROM's own in-span predictions through an **incremental
SVD (iSVD) with forgetting** reweights and rotates the basis inside the current
subspace, preparing it to absorb the next external correction far more
effectively. The projection is a **Least-Squares Petrov-Galerkin (LSPG)** ROM
with QDEIM hyper-reduction.

This is the companion code to our paper:

> *In-span learning: adapting reduced-order models using their own predictions.* <br>
> Amirpasha Hedayat, Laura Balzano, Karthik Duraisamy. <br>
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) *(link will be updated on posting)*

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)).

---

## The idea in one minute

A reduced-order model compresses an expensive high-dimensional simulation into a
small subspace and evolves only a few coordinates inside it — fast, but it loses
accuracy once the dynamics drift beyond the training data.

**Adaptive ROMs** fix this by updating the subspace online using *external*
information — an occasional full-order solve or sensor snapshot. Such information
lies **outside** the current subspace, so we call it **out-of-span**. This is the
standard, state-of-the-art recipe (our *baseline adaptive ROM*).

Between those external corrections, the ROM constantly produces predictions of
its own. By construction these lie **inside** the current subspace — they are
**in-span** — so the usual subspace-update view discards them. We show that is
wrong: streamed through an iSVD with forgetting, the ROM's own in-span
predictions **rotate and reweight the basis inside the subspace**. They don't
move the subspace; they change how it is *prepared* to absorb the next external
correction.

```
  baseline adaptive ROM:   predict ─ predict ─ predict ─ CORRECT ─ predict ─ ...
                                                          ▲ external (out-of-span)

  SPIN ROM:                predict ─ in-span ─ in-span ─ CORRECT ─ in-span ─ ...
                                     ▲ learn from   ▲ same external correction,
                                       own output     but absorbed better
```

**SPIN is exactly the baseline adaptive ROM, plus an in-span update between every
pair of external corrections** — same correction budget, no extra full-order
solves, just more use of information the model already generated.

> SPIN's out-of-span channel (the baseline adaptive ROM we compare against) is
> itself a strong, recent method — see our previous work,
> [arXiv:2605.28684](https://arxiv.org/abs/2605.28684). SPIN adds the in-span
> channel on top of it.

---

## Installation

```bash
git clone https://github.com/USERNAME/SPIN.git
cd SPIN
pip install -e .
```

Python ≥ 3.9 with NumPy, SciPy, and Matplotlib (see [`requirements.txt`](requirements.txt)).
The demo notebooks also need Jupyter:

```bash
pip install -e ".[notebooks]"
```

---

## Quick start

```python
from spin.core import compute_pod_basis, qdeim
from spin.problems.burgers import BurgersSolver, BurgersSpinROM
from spin.diagnostics import relative_l2_error

# 1) Run the FOM to get training + reference snapshots.
fom = BurgersSolver(Nx=256, L=1.0, nu=1e-2, dt=1e-3)
snaps = fom.simulate(fom.initial_condition(), n_steps=500, tol=1e-8)  # (Nx, 501)

# 2) Offline basis + QDEIM indices from a handful of early snapshots.
Phi, sigma = compute_pod_basis(snaps[:, :4], r=4)
p_inds = qdeim(Phi, n_sensors=4)
a0 = Phi.T @ snaps[:, 0]

# 3) The same driver gives all three models via `mode`.
def run(mode, gamma_in=1.0, gamma_out=0.25):
    rom = BurgersSpinROM(Phi.copy(), sigma.copy(), p_inds.copy(),
                         dt=1e-3, nu=1e-2, dx=fom.dx, zs=10,
                         mode=mode, gamma_in=gamma_in, gamma_out=gamma_out)
    return rom.simulate(a0.copy(), n_steps=500)

baseline = run("baseline", gamma_out=0.01)
spin     = run("spin",     gamma_in=1.0, gamma_out=0.25)

print("baseline adaptive:", relative_l2_error(baseline, snaps).mean())  # ~5.5e-2
print("SPIN             :", relative_l2_error(spin,     snaps).mean())  # ~1.9e-2
```

The only thing that changes between the two runs is `mode`:

| `mode` | in-span updates | out-of-span corrections | what it is |
|---|:---:|:---:|---|
| `"static"`   | ✗ | ✗ | fixed POD basis, no adaptation |
| `"baseline"` | ✗ | ✓ | baseline adaptive ROM (state-of-the-art) |
| `"spin"`     | ✓ | ✓ | **SPIN** (this work) |

**Key knobs:** `zs` (how often an out-of-span correction is taken), `gamma_out`
(out-of-span forgetting), `gamma_in` (in-span forgetting — the memory horizon of
in-span learning).

See [`examples/`](examples) for fully worked, runnable notebooks.

---

## The method, in one picture

Every `zs` steps, SPIN performs a single full-order operator query to obtain a
correction snapshot; this **out-of-span** signal updates the basis via iSVD
(forgetting `gamma_out`), then the QDEIM indices are refreshed. On *every other*
step, SPIN instead feeds the ROM's **own prediction** through the same iSVD
(forgetting `gamma_in`) — an **in-span** update that rotates and reweights the
basis without moving the subspace. The full procedure is Algorithm 1 in the paper.

```
      offline               online (loop)
   ┌───────────┐   ┌──────────────────────────────────────────────────────┐
   │ FOM snaps │──▶│ LSPG step (Newton)                                     │
   └───────────┘   │        │                                              │
                   │        ▼                                              │
   ┌───────────┐   │  every zs steps:   one FOM query ─▶ OUT-OF-SPAN iSVD  │
   │ POD basis │──▶│                                     (moves subspace)   │
   └───────────┘   │  otherwise:        own prediction ─▶ IN-SPAN iSVD      │
   ┌───────────┐   │                                     (rotates basis)    │
   │   QDEIM   │──▶│        │                                              │
   └───────────┘   │        ▼                                              │
                   │  refresh QDEIM + reproject coords ────────────────────┘
                   └──────────────────────────────────────────────────────┘
```

---

## Package layout

```
SPIN/
├── src/spin/
│   ├── core/                  # problem-agnostic building blocks
│   │   ├── pod.py             #   POD via thin SVD
│   │   ├── sampling.py        #   QDEIM index selection
│   │   ├── basis_adaptation.py#   iSVD with forgetting (the in-span / out-of-span update)
│   │   ├── rom_base.py        #   LSPGROMBase: generic static LSPG+QDEIM ROM
│   │   └── adaptive_rom.py    #   SpinROMBase: generic static / baseline / SPIN loop
│   ├── problems/              # reference equations (thin templates)
│   │   ├── spiral/            #   closed-form spiral: FOM + experiment
│   │   ├── burgers/           #   1D viscous Burgers: FOM + ROM
│   │   └── fisher_kpp/        #   1D Fisher-KPP: FOM + ROM
│   ├── diagnostics.py         # the metrics used in the paper's figures
│   └── utils/                 # plotting helpers + shared palette
├── examples/
│   ├── 01_spiral.ipynb        # the toy example that exposes the mechanism
│   ├── 02_burgers.ipynb       # viscous Burgers, long-horizon prediction
│   └── 03_fisher_kpp.ipynb    # Fisher-KPP fronts, long-horizon prediction
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

The **core** modules are entirely problem-agnostic and reusable across equations.
The **problems** modules are concrete templates showing how to couple a specific
equation's hyper-reduced residual and Jacobian to the core machinery.

---

## Bringing SPIN to your own equation

All problem-specific code lives behind two methods. To run SPIN on a new
equation you only need to:

1. **Provide a FOM solver** exposing a `.step(u, tol, max_iter, verbose) -> u_new`
   method that advances your full state by one time step (for the out-of-span
   correction query). Any implicit solver you already have will do.
2. **Subclass `SpinROMBase`** (via a small mixin, as the reference problems do)
   and implement:
   - `residual_sample(self, a, a_old) -> (m,) ndarray` — the LSPG residual at the
     sampling indices `self.p_inds`, evaluated at `u = self.Phi @ a`;
   - `jacobian_sample(self, a) -> (m, r) ndarray` — the corresponding sampled LSPG
     test operator `W = d(residual)/da`.

```python
import numpy as np
from spin.core.adaptive_rom import SpinROMBase

class MyEquationSpinROM(SpinROMBase):
    def residual_sample(self, a, a_old):
        u = self.Phi @ a
        # ... evaluate your sampled spatial operator F at rows self.p_inds ...
        return self._Phi_p @ (a - a_old) + self.dt * Fp

    def jacobian_sample(self, a):
        u = self.Phi @ a
        # ... rows of d(residual)/da at the sample points ...
        return self._Phi_p + self.dt * JfPhi   # (m, r)
```

The base class provides the cached sampling data you need inside these hooks:
`self._Phi_p`, `self._Phi_up`, `self._Phi_down` (the basis at the sample points
and their periodic neighbours), `self.p_inds`, and `self.M` (the QDEIM
pseudoinverse), all refreshed automatically after every adaptation event. Then:

```python
rom = MyEquationSpinROM(
    Phi=Phi, sigma=sigma, p_inds=p_inds, dt=dt,
    fom_solver=my_fom, zs=10,
    mode="spin", gamma_in=1.0, gamma_out=0.25,
)
u_pred = rom.simulate(a0, n_steps=500)   # (N, n_steps+1)
```

`src/spin/problems/burgers/rom.py` and `src/spin/problems/fisher_kpp/rom.py` are
concrete, commented templates demonstrating both hooks in practice (including the
static ROM via `LSPGROMBase`).

---

## Reproducing the paper

Open the notebooks in [`examples/`](examples) and run them top to bottom. Each one
has all of its control variables in a single cell near the top, so you can
immediately change the rank, correction interval, or forgetting factors and see
the effect. With the default values they reproduce the main-text results:

| experiment | static | baseline adaptive | SPIN |
|---|---|---|---|
| **Burgers** (mean rel. $L_2$) | 3.7e-1 | 5.5e-2 | **1.9e-2** |
| **Fisher-KPP** (mean rel. $L_2$) | 2.1e-1 | 1.2e-1 | **7.8e-3** |

For the spiral, the notebook reproduces the reported residual capture
(0.26 → 0.80), plane-change angle (15° → 53°), and correction error
(1.0e-2 → 8.8e-4) when in-span preconditioning is enabled.

---

## License

[MIT](LICENSE).

## Citation

A BibTeX entry will be finalized upon publication; see [`CITATION.cff`](CITATION.cff).

```bibtex
@article{hedayat2026inspan,
  title   = {In-span learning: adapting reduced-order models using their own predictions},
  author  = {Hedayat, Amirpasha and Balzano, Laura and Duraisamy, Karthik},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
