"""Microbenchmarks for hot paths in pymatgen.core.

Run manually:
    uv run python tests/performance/bench_core.py

This script is intentionally standalone (no pytest, no xdist) to give
stable timings. Each benchmark is calibrated to run for ~0.2 s and reports
mean ns/call across multiple repeats.
"""

from __future__ import annotations

import timeit
import warnings
from typing import TYPE_CHECKING

import numpy as np

from pymatgen.core import Composition, Element, Lattice, Species, Structure
from pymatgen.core.periodic_table import get_el_sp

if TYPE_CHECKING:
    from collections.abc import Callable

# Silence Element.X NaN warnings so they do not skew the X-accessing benches.
warnings.filterwarnings("ignore", message=r"No Pauling electronegativity.*")


def _time(stmt: Callable[[], object], *, target_s: float = 0.2, repeats: int = 5) -> tuple[float, float]:
    """Return (mean ns/call, stdev ns/call) over `repeats` runs.

    Calibrates `number` so each run lasts ~target_s.
    """
    # Calibration: find a `number` that takes >= ~10 ms.
    number = 1
    while True:
        t = timeit.timeit(stmt, number=number)
        if t >= 0.01:
            break
        number *= 10
    # Now scale to target_s.
    number = max(1, int(number * (target_s / max(t, 1e-9))))
    samples = [timeit.timeit(stmt, number=number) / number for _ in range(repeats)]
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / len(samples)
    stdev = var**0.5
    return mean * 1e9, stdev * 1e9


def _fmt(ns: float, sd: float) -> str:
    if ns >= 1e6:
        return f"{ns / 1e6:8.2f} ms  ± {sd / 1e6:.2f}"
    if ns >= 1e3:
        return f"{ns / 1e3:8.2f} µs  ± {sd / 1e3:.2f}"
    return f"{ns:8.1f} ns  ± {sd:.1f}"


def run(name: str, stmt: Callable[[], object]) -> None:
    ns, sd = _time(stmt)
    print(f"  {name:<48s} {_fmt(ns, sd)}")


def main() -> None:
    print("=" * 80)
    print("pymatgen.core microbenchmarks")
    print("=" * 80)

    # --- Composition parsing ------------------------------------------------
    print("\n[Composition] formula parsing")
    formulas_simple = ("LiFePO4", "Fe2O3", "NaCl", "H2O")
    formulas_complex = ("Li3Fe2(PO4)3", "Y3N@C80", "(NH4)2[PtCl6]", "Ca5(PO4)3F")

    def _parse_simple() -> None:
        for f in formulas_simple:
            Composition(f)

    def _parse_complex() -> None:
        for f in formulas_complex:
            Composition(f)

    run("Composition(simple) x4", _parse_simple)
    run("Composition(nested-parens) x4", _parse_complex)

    # Same formula repeated — measures benefit of caching.
    def _parse_repeat() -> None:
        for _ in range(8):
            Composition("LiFePO4")

    run("Composition('LiFePO4') x8 (repeat)", _parse_repeat)

    # --- get_el_sp ----------------------------------------------------------
    print("\n[get_el_sp] dispatch")
    run("get_el_sp('Fe')", lambda: get_el_sp("Fe"))
    run("get_el_sp('Fe2+')", lambda: get_el_sp("Fe2+"))
    run("get_el_sp(26)", lambda: get_el_sp(26))
    el_fe = Element("Fe")
    run("get_el_sp(Element('Fe'))", lambda: get_el_sp(el_fe))

    # --- Element properties -------------------------------------------------
    print("\n[Element] property access")
    fe = Element("Fe")
    he = Element("He")  # has no Pauling X

    run("Fe.X (200x)", lambda: [fe.X for _ in range(200)])
    run("He.X NaN path (200x)", lambda: [he.X for _ in range(200)])
    run("Fe.Z (200x)", lambda: [fe.Z for _ in range(200)])
    run("Fe.atomic_mass (200x)", lambda: [fe.atomic_mass for _ in range(200)])
    run("Fe.ionic_radii (50x)", lambda: [fe.ionic_radii for _ in range(50)])
    run("Fe.average_ionic_radius (50x)", lambda: [fe.average_ionic_radius for _ in range(50)])
    run("Fe.atomic_orbitals_eV (50x)", lambda: [fe.atomic_orbitals_eV for _ in range(50)])
    run("Fe.full_electronic_structure (50x)", lambda: [fe.full_electronic_structure for _ in range(50)])
    run("Fe.row (200x)", lambda: [fe.row for _ in range(200)])
    run("Fe.group (200x)", lambda: [fe.group for _ in range(200)])
    run("Fe.iupac_ordering (200x)", lambda: [fe.iupac_ordering for _ in range(200)])
    run("Fe.thermal_conductivity __getattr__ (200x)", lambda: [fe.thermal_conductivity for _ in range(200)])

    # --- Composition properties --------------------------------------------
    print("\n[Composition] derived properties")
    comp = Composition("Li3Fe2(PO4)3")
    big = Composition({Element(s): 1 for s in ("Li", "Na", "K", "Mg", "Ca", "Sr", "Ba", "Fe", "Co", "Ni", "O", "S")})

    run("comp.average_electroneg (200x)", lambda: [comp.average_electroneg for _ in range(200)])
    run("comp.total_electrons (200x)", lambda: [comp.total_electrons for _ in range(200)])
    run("comp.num_atoms (200x)", lambda: [comp.num_atoms for _ in range(200)])
    run("comp.weight (200x)", lambda: [comp.weight for _ in range(200)])
    run("comp.formula (200x)", lambda: [comp.formula for _ in range(200)])
    run("comp.reduced_formula (200x)", lambda: [comp.reduced_formula for _ in range(200)])
    run("big.average_electroneg (200x)", lambda: [big.average_electroneg for _ in range(200)])

    # --- Sorting compositions (uses Element.X and Composition.average_electroneg)
    print("\n[Composition] sorting (Element.X heavy)")
    comps = [Composition(f) for f in ("LiFePO4", "NaCl", "Fe2O3", "Al2O3", "MgO", "SiO2", "TiO2", "CaCO3")]
    run("sorted(comps) x10", lambda: [sorted(comps) for _ in range(10)])

    # --- Species creation ---------------------------------------------------
    print("\n[Species]")
    run("Species('Fe', 2)", lambda: Species("Fe", 2))
    run("Species.from_str('Fe2+')", lambda: Species.from_str("Fe2+"))

    # --- Structure construction + coord access ------------------------------
    print("\n[Structure] construction and coord access")
    # A representative ordered crystal: NaCl 4x4x4 supercell (256 sites).
    rng = np.random.default_rng(0)
    lattice = Lattice.cubic(5.64)
    base_species = ["Na", "Cl", "Na", "Cl", "Na", "Cl", "Na", "Cl"]
    base_frac = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ]
    nacl = Structure(lattice, base_species, base_frac)
    big_lattice = Lattice.cubic(5.64 * 4)
    big_species = base_species * 64
    big_frac = (np.tile(np.asarray(base_frac), (64, 1)) + rng.uniform(0, 1, (512, 3))) % 1.0
    big_species = big_species[:512]

    # Cart-coord constructor (forces frac->cart batch conversion).
    big_cart = big_lattice.get_cartesian_coords(big_frac)
    run("Structure(NaCl, N=8) from frac", lambda: Structure(lattice, base_species, base_frac))
    run("Structure(N=512) from frac", lambda: Structure(big_lattice, big_species, big_frac))
    run("Structure(N=512) from cart", lambda: Structure(big_lattice, big_species, big_cart, coords_are_cartesian=True))

    big = Structure(big_lattice, big_species, big_frac)
    run("big.cart_coords (50x)", lambda: [big.cart_coords for _ in range(50)])
    run("big.frac_coords (50x)", lambda: [big.frac_coords for _ in range(50)])
    run("big.distance_matrix (5x)", lambda: [big.distance_matrix for _ in range(5)])
    run("big.copy() (5x)", lambda: [big.copy() for _ in range(5)])
    run("nacl * (2,2,2) supercell (5x)", lambda: [nacl * (2, 2, 2) for _ in range(5)])


if __name__ == "__main__":
    main()
