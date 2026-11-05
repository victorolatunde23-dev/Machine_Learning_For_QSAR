"""Reaction-based enumeration of a focused virtual library.

A reaction SMARTS that is wrong does not raise. It silently matches nothing, or builds the
wrong regiochemistry, and the enumeration quietly shrinks or fills with nonsense. So every
reaction here ships a ``(reactants -> expected product)`` self-test, the tests run at load
time, and **a reaction that fails is disabled rather than trusted**. That is not
defensive padding: it is how the click-chemistry template in this package was caught
producing nothing at all.

Two further guards, both from real failures:

* A validated reaction with no matching building block contributes nothing and used to
  vanish silently -- the attempt counter only covers combinations that were *tried*.
  :func:`enumerate_library` reports reactions that had no eligible blocks.
* The property window is drawn from the known actives, deliberately not from Lipinski
  (PSMA-617 is MW ~1042 with eight carboxylic acids, so a rule-of-five filter would delete
  the entire active class). But a window drawn from actives can still exclude nearly every
  product, because a product is a core *plus* a cap. Applied as a hard filter with no floor
  it emptied a library outright, so below ``min_keep_frac`` it flags instead of filtering.

The enumeration is also **circular by construction**: building analogues around the
training pharmacophore makes them in-domain by definition and the model partly recognises
its own training distribution. :func:`positive_control` is the check that keeps it honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._optional import require, tqdm

__all__ = [
    "Reaction",
    "BuildingBlock",
    "ReactionLibrary",
    "validate_reactions",
    "enumerate_library",
    "property_window",
    "apply_property_window",
    "positive_control",
    "DEFAULT_REACTIONS",
]


@dataclass
class Reaction:
    """A reaction template with the self-test that proves it works."""

    name: str
    smarts: str
    # Roles, in the order the SMARTS expects its reactants.
    roles: Tuple[str, ...]
    # (reactant SMILES in SMARTS order, expected product SMILES)
    test: Tuple[Tuple[str, ...], str]
    note: str = ""

    def compile(self):
        rdChemReactions = require("rdkit").Chem.rdChemReactions  # type: ignore
        rxn = rdChemReactions.ReactionFromSmarts(self.smarts)
        if rxn is None:
            raise ValueError(f"{self.name}: SMARTS failed to parse")
        rxn.Initialize()
        return rxn


@dataclass
class BuildingBlock:
    """One core or cap, with the role that decides which reactions it can enter."""

    id: str
    smiles: str
    role: str
    note: str = ""


# The Glu-urea templates below cover the couplings used most often to decorate a
# zinc-binding core. Roles are (cap_role, core_role).
DEFAULT_REACTIONS: Tuple[Reaction, ...] = (
    Reaction(
        name="amide_coupling",
        smarts="[C:1](=[O:2])[OX2H1].[NX3;H2,H1;!$(N-[C,S]=[O,S,N]):3]>>[C:1](=[O:2])[N:3]",
        roles=("acid", "amine"),
        test=(("CC(=O)O", "CN"), "CNC(C)=O"),
        note="carboxylic acid + amine",
    ),
    Reaction(
        name="sulfonamide",
        smarts="[S:1](=[O:2])(=[O:3])[Cl].[NX3;H2,H1;!$(N-[C,S]=[O,S,N]):4]"
        ">>[S:1](=[O:2])(=[O:3])[N:4]",
        roles=("sulfonyl_chloride", "amine"),
        test=(("CS(=O)(=O)Cl", "CN"), "CNS(C)(=O)=O"),
    ),
    Reaction(
        name="urea_from_isocyanate",
        smarts="[N:1]=[C:2]=[O:3].[NX3;H2,H1;!$(N-[C,S]=[O,S,N]):4]"
        ">>[N:1][C:2](=[O:3])[N:4]",
        roles=("isocyanate", "amine"),
        test=(("CN=C=O", "CN"), "CNC(=O)NC"),
    ),
    Reaction(
        name="thiourea_from_isothiocyanate",
        smarts="[N:1]=[C:2]=[S:3].[NX3;H2,H1;!$(N-[C,S]=[O,S,N]):4]"
        ">>[N:1][C:2](=[S:3])[N:4]",
        roles=("isothiocyanate", "amine"),
        test=(("CN=C=S", "CN"), "CNC(=S)NC"),
    ),
    Reaction(
        name="reductive_amination",
        smarts="[CX3H1:1]=[OX1].[NX3;H2,H1;!$(N-[C,S]=[O,S,N]):2]>>[CH2:1][N:2]",
        roles=("aldehyde", "amine"),
        test=(("CC=O", "CN"), "CCNC"),
    ),
    Reaction(
        name="snar_amination",
        smarts="[c:1][F,Cl,Br,I].[NX3;H2,H1;!$(N-[C,S]=[O,S,N]):2]>>[c:1][N:2]",
        roles=("aryl_halide", "amine"),
        test=(("Clc1ccccc1", "CN"), "CNc1ccccc1"),
    ),
    Reaction(
        name="click_triazole",
        # The charges matter. A mapped product atom inherits its reactant's formal charge
        # unless the template says otherwise, so without the explicit +0/-0 the azide's
        # +1/-1 carry into the triazole, the ring valences break, sanitisation fails and
        # the reaction silently yields nothing. That is exactly what the self-test caught.
        smarts="[C;H1:1]#[C:2].[N:3]=[N+:4]=[N-:5]"
        ">>[N:3]1[N+0:4]=[N-0:5][C:2]=[C:1]1",
        roles=("alkyne", "azide"),
        test=(("OCC#C", "C(c1ccccc1)N=[N+]=[N-]"), "OCc1cn(Cc2ccccc2)nn1"),
        note="CuAAC, 1,4-disubstituted; how most tracer chemistry attaches a chelator",
    ),
)


# ------------------------------------------------------------------------------ validation


@dataclass
class ReactionLibrary:
    """Reactions that passed their self-tests, and the ones that did not."""

    active: Dict[str, Reaction] = field(default_factory=dict)
    compiled: Dict[str, object] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.active)

    @property
    def roles_needed(self) -> Dict[str, Tuple[str, ...]]:
        return {name: rxn.roles for name, rxn in self.active.items()}

    def summary(self, total: Optional[int] = None) -> str:
        total = total if total is not None else len(self.active) + len(self.failures)
        lines = ["REACTION SELF-TESTS"]
        for name in self.active:
            lines.append(f"  PASS  {name}")
        for name, why in self.failures.items():
            lines.append(f"  FAIL  {name:<30s} {why}")
        lines.append(
            f"\n{len(self.active)} of {total} reactions validated and available for "
            f"enumeration."
        )
        if self.failures:
            lines += [
                "  Failed reactions are DISABLED -- their SMARTS needs fixing before they",
                "  can be trusted. A silently wrong template is worse than a missing one.",
            ]
        return "\n".join(lines)


def _canonical(smiles: str) -> Optional[str]:
    Chem = require("rdkit").Chem  # type: ignore[attr-defined]
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def _run_reaction(rxn, reactant_mols: Sequence) -> List[str]:
    """Product SMILES from one reaction application, sanitised, deduplicated, ordered."""
    Chem = require("rdkit").Chem  # type: ignore[attr-defined]
    out: List[str] = []
    try:
        product_sets = rxn.RunReactants(tuple(reactant_mols))
    except Exception:
        return out
    for products in product_sets:
        for product in products:
            try:
                Chem.SanitizeMol(product)
                smi = Chem.MolToSmiles(product)
            except Exception:
                continue
            if smi and smi not in out:
                out.append(smi)
    return out


def validate_reactions(
    reactions: Iterable[Reaction] = DEFAULT_REACTIONS, *, verbose: bool = True
) -> ReactionLibrary:
    """Run every reaction's self-test; keep the ones that reproduce their product.

    Raises if RDKit is absent rather than reporting every reaction as failed. The whole
    point of a FAIL here is "this SMARTS is wrong", and a missing toolkit reported the
    same way would destroy that signal.
    """
    require("rdkit")  # fail once, clearly, instead of once per reaction
    library = ReactionLibrary()
    reactions = list(reactions)
    for rxn_def in reactions:
        try:
            rxn = rxn_def.compile()
        except Exception as exc:
            library.failures[rxn_def.name] = f"{type(exc).__name__}: {exc}"
            continue
        reactant_smiles, expected = rxn_def.test
        Chem = require("rdkit").Chem  # type: ignore[attr-defined]
        mols = [Chem.MolFromSmiles(s) for s in reactant_smiles]
        if any(m is None for m in mols):
            library.failures[rxn_def.name] = "a test reactant did not parse"
            continue
        got = _run_reaction(rxn, mols)
        if not got:
            library.failures[rxn_def.name] = "produced nothing"
            continue
        want = _canonical(expected)
        if want is not None and want not in {_canonical(g) for g in got}:
            library.failures[rxn_def.name] = (
                f"produced {got[0]} but the test expects {expected}"
            )
            continue
        library.active[rxn_def.name] = rxn_def
        library.compiled[rxn_def.name] = rxn
    if verbose:
        print(library.summary(total=len(reactions)))
    return library


# ---------------------------------------------------------------------------- enumeration


def enumerate_library(
    cores: Sequence[BuildingBlock],
    caps: Sequence[BuildingBlock],
    library: ReactionLibrary,
    *,
    max_products: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Cross product of (core x reaction x cap), matched on roles.

    Every row carries its ``core_id``, ``reaction``, ``cap_id``, ``cap_role`` and
    ``variant``, so a hit is a synthesis proposal rather than only a structure. Extra
    regiochemistries are kept as separate variants and deduplicated later: they are
    genuine alternative products, not errors.
    """
    Chem = require("rdkit").Chem  # type: ignore[attr-defined]

    core_mols: Dict[str, Tuple[BuildingBlock, object]] = {}
    for block in cores:
        mol = Chem.MolFromSmiles(block.smiles)
        if mol is None:
            if verbose:
                print(f"  !! core {block.id!r} did not parse; skipped")
            continue
        core_mols[block.id] = (block, mol)

    cap_mols: Dict[str, Tuple[BuildingBlock, object]] = {}
    for block in caps:
        mol = Chem.MolFromSmiles(block.smiles)
        if mol is None:
            if verbose:
                print(f"  !! cap {block.id!r} did not parse; skipped")
            continue
        cap_mols[block.id] = (block, mol)

    combos = [
        (name, cap_id, core_id)
        for name, rxn_def in library.active.items()
        for cap_id in cap_mols
        for core_id in core_mols
        if cap_mols[cap_id][0].role == rxn_def.roles[0]
        and core_mols[core_id][0].role == rxn_def.roles[1]
    ]

    rows: List[Dict[str, object]] = []
    n_empty = 0
    it = tqdm(combos, desc="Enumerating", unit="combo") if verbose else combos
    for name, cap_id, core_id in it:
        rxn = library.compiled[name]
        cap, cap_mol = cap_mols[cap_id]
        core, core_mol = core_mols[core_id]
        products = _run_reaction(rxn, (cap_mol, core_mol))
        if not products:
            n_empty += 1
            continue
        for variant, smi in enumerate(products):
            rows.append(
                {
                    "smiles": smi,
                    "core_id": core_id,
                    "reaction": name,
                    "cap_id": cap_id,
                    "cap_role": cap.role,
                    "variant": variant,
                }
            )

    frame = pd.DataFrame(rows)
    if verbose:
        print(f"\n  {len(combos)} core x reaction x cap combinations attempted")
        print(
            f"  {n_empty} produced nothing "
            f"({100 * n_empty / max(len(combos), 1):.0f}%)"
        )
        print(f"  {len(frame):,} raw products before deduplication")
        # A validated reaction with no eligible building block contributes nothing and
        # would otherwise vanish: the count above only covers combinations that were tried.
        used = {name for name, _, _ in combos}
        unused = {n: r.roles for n, r in library.active.items() if n not in used}
        if unused:
            present = sorted({b.role for b in caps} | {b.role for b in cores})
            print(
                f"  {len(unused)} validated reaction(s) contributed NOTHING, because no "
                f"building block carries the role they need:"
            )
            for name, roles in unused.items():
                print(
                    f"    {name:<30s} needs cap role {roles[0]!r} + core role {roles[1]!r}"
                )
            print(f"    roles actually present: {', '.join(present)}")
            print("    Add blocks with those roles, or drop the reaction.")

    if max_products and len(frame) > max_products:
        frame = frame.sample(max_products, random_state=random_state).reset_index(drop=True)
        if verbose:
            print(f"  sampled down to {max_products:,} products")

    if len(frame):
        frame["compound_id"] = [
            f"{r.core_id}|{r.reaction}|{r.cap_id}" + (f"|v{r.variant}" if r.variant else "")
            for r in frame.itertuples(index=False)
        ]
    return frame


# ------------------------------------------------------------------------ property window


PROPERTY_FUNCTIONS = ("MolWt", "TPSA", "NumHDonors", "NumRotatableBonds")


def compute_properties(smiles: Sequence[str], *, progress: bool = False) -> pd.DataFrame:
    """The four bulk properties the window is defined over."""
    Chem = require("rdkit").Chem  # type: ignore[attr-defined]
    Descriptors = require("rdkit").Chem.Descriptors  # type: ignore[attr-defined]
    fns = {name: getattr(Descriptors, name) for name in PROPERTY_FUNCTIONS}
    rows = []
    it = tqdm(smiles, desc="Properties", unit="mol", leave=False) if progress else smiles
    for smi in it:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            rows.append({k: np.nan for k in fns})
            continue
        row = {}
        for name, fn in fns.items():
            try:
                row[name] = float(fn(mol))
            except Exception:
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=list(fns))


def property_window(
    active_smiles: Sequence[str], *, quantile: float = 0.05, min_actives: int = 10
) -> Optional[pd.DataFrame]:
    """Low/high bounds per property, from the known actives' quantiles.

    Drawn from the actives rather than from a rule of five on purpose: on this target the
    actives run to MW ~1042 with eight carboxylic acids, so Lipinski would delete the
    entire positive class.
    """
    props = compute_properties(active_smiles).dropna()
    if len(props) < min_actives:
        return None
    return pd.DataFrame(
        {
            "low": props.quantile(quantile),
            "high": props.quantile(1.0 - quantile),
            "median": props.median(),
        }
    )


def apply_property_window(
    frame: pd.DataFrame,
    window: Optional[pd.DataFrame],
    *,
    apply: bool = True,
    min_keep_frac: float = 0.20,
    verbose: bool = True,
) -> pd.DataFrame:
    """Filter or flag products against the window, with a floor on the filter.

    Below ``min_keep_frac`` the window is reported and the products are flagged in
    ``in_property_window`` rather than removed. An enumerated product is a core *plus* a
    cap, so it is systematically heavier and more polar than the actives the window came
    from; applying the filter regardless once emptied a library completely and made every
    number downstream vacuous.
    """
    if window is None:
        if verbose:
            print("\n  Too few known actives to derive a property window; none applied.")
        return frame
    missing = [c for c in window.index if c not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing property columns {missing}")

    keep = np.ones(len(frame), dtype=bool)
    for prop in window.index:
        keep &= frame[prop].between(window.loc[prop, "low"], window.loc[prop, "high"]).values
    frac = float(keep.mean()) if len(keep) else 0.0

    if verbose:
        print(f"\n  Property window from the known actives:")
        for prop in window.index:
            print(
                f"    {prop:<18s} {window.loc[prop, 'low']:9.1f} .. "
                f"{window.loc[prop, 'high']:9.1f}   "
                f"(actives median {window.loc[prop, 'median']:.1f})"
            )
        print(
            f"  property filter: {len(frame):,} -> {int(keep.sum()):,} products "
            f"({100 * frac:.0f}% kept)"
        )

    if not apply:
        out = frame.copy()
        out["in_property_window"] = keep
        if verbose:
            print("  (apply is False -- flagged, not removed)")
        return out
    if frac < min_keep_frac:
        out = frame.copy()
        out["in_property_window"] = keep
        if verbose:
            print(
                f"\n  !! The window would keep only {100 * frac:.0f}% of products, below "
                f"the {100 * min_keep_frac:.0f}% floor, so it has NOT been applied -- the "
                f"products are flagged in `in_property_window` instead."
            )
            print(
                "     This is the expected direction: a product is a core plus a cap, so "
                "it is\n     heavier and more polar than the actives the window came "
                "from. Either widen\n     the window, or treat the flag as a soft "
                "preference. An empty library would\n     have made every result below "
                "meaningless."
            )
        return out
    return frame[keep].reset_index(drop=True)


# ------------------------------------------------------------------------ positive control


def positive_control(
    product_keys: Sequence[Optional[str]],
    training_keys: set,
    *,
    skeleton_keys: Optional[set] = None,
) -> Dict[str, object]:
    """Which enumerated products reproduce a known training compound.

    Checked on two tiers. The full InChIKey encodes stereochemistry, tautomer and
    protonation state; the 14-character connectivity block does not. That distinction is
    load-bearing: a full-key-only control once reported "0 products match a training
    compound" -- and told the user the shortlist was unfalsifiable -- while two products
    had Tanimoto distance exactly 0.000 and descriptor distance ~1e-13 to a training
    structure. They were the same constitution, differing only in a layer the achiral
    fingerprint ignores.
    """
    from ..features.molecules import skeleton_key

    exact = [bool(k and k in training_keys) for k in product_keys]
    skeletons = skeleton_keys
    if skeletons is None:
        skeletons = {skeleton_key(k) for k in training_keys if k}
    loose = [
        bool(k and skeleton_key(k) in skeletons) for k in product_keys
    ]
    return {
        "exact": np.asarray(exact, dtype=bool),
        "connectivity": np.asarray(loose, dtype=bool),
        "n_exact": int(sum(exact)),
        "n_connectivity": int(sum(loose)),
    }
