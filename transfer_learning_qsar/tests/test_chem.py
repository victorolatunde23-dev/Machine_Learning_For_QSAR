"""Tests that need a chemistry toolkit. Skipped cleanly when RDKit is absent.

They are worth keeping even in an environment without RDKit: run the suite where it *is*
installed and these are the ones that check the reaction SMARTS, which cannot be verified
any other way -- a wrong template does not raise, it silently yields nothing.
"""

import numpy as np
import pandas as pd
import pytest

rdkit = pytest.importorskip("rdkit", reason="needs the 'chem' extra")

pytestmark = pytest.mark.chem


# ---------------------------------------------------------------------------- featurising


def test_standardiser_strips_salts_and_charges():
    from qsar_tl.features.molecules import Standardiser

    std = Standardiser()
    from rdkit import Chem

    mol = std.from_smiles("CC(=O)[O-].[Na+]")
    assert mol is not None
    out = Chem.MolToSmiles(mol)
    assert "Na" not in out          # largest fragment only
    assert "-]" not in out          # uncharged


def test_standardiser_returns_none_on_nonsense():
    from qsar_tl.features.molecules import Standardiser

    std = Standardiser()
    assert std.from_smiles("not a molecule at all") is None
    assert std.from_smiles("") is None
    assert std.from_smiles(None) is None


def test_morgan_fingerprint_size_and_determinism():
    from qsar_tl.features.molecules import fingerprints_to_array, morgan_fingerprint
    from rdkit import Chem

    mol = Chem.MolFromSmiles("c1ccccc1C(=O)NCCN")
    a = fingerprints_to_array([morgan_fingerprint(mol, 2, 512)])
    b = fingerprints_to_array([morgan_fingerprint(mol, 2, 512)])
    assert a.shape == (1, 512)
    assert np.array_equal(a, b)


def test_skeleton_key_ignores_stereochemistry():
    """The two-tier positive control depends on this.

    A full-InChIKey comparison once reported no overlap between an enumerated library and
    the training set while two products had Tanimoto distance exactly 0.000 to a training
    structure: same constitution, different stereo layer.
    """
    from qsar_tl.features.molecules import inchikey, skeleton_key
    from rdkit import Chem

    a = inchikey(Chem.MolFromSmiles("N[C@@H](CCC(=O)O)C(=O)O"))
    b = inchikey(Chem.MolFromSmiles("N[C@H](CCC(=O)O)C(=O)O"))
    assert a != b
    assert skeleton_key(a) == skeleton_key(b)


def test_featurise_drops_unparseable_and_dedupes():
    from qsar_tl.features.molecules import featurise

    frame = pd.DataFrame(
        {"smiles": ["CCO", "CCO", "c1ccccc1", "!!!bad!!!"], "y_pKi": [1, 1, 0, 0]}
    )
    fs = featurise(frame, n_bits=256, progress=False, dedupe=True)
    assert fs.n_failed == 1
    assert fs.n_duplicates == 1
    assert len(fs) == 2
    assert len(fs.fingerprints) == 2


def test_design_matrix_shapes_per_feature_set():
    from qsar_tl.features.molecules import build_design_matrix, featurise

    frame = pd.DataFrame({"smiles": ["CCO", "c1ccccc1", "CCN"]})
    fs = featurise(frame, n_bits=128, with_descriptors=True, progress=False)
    fp_only = build_design_matrix(fs.fingerprints, fs.descriptors, feature_set="fp_only")
    both = build_design_matrix(
        fs.fingerprints, fs.descriptors, feature_set="fp_plus_descriptors"
    )
    assert fp_only.shape == (3, 128)
    assert both.shape[1] == 128 + fs.descriptors.shape[1]


def test_descriptors_only_without_descriptors_raises():
    from qsar_tl.features.molecules import build_design_matrix

    with pytest.raises(ValueError, match="needs descriptors"):
        build_design_matrix([], None, feature_set="descriptors_only")


def test_descriptor_selector_reduces_redundancy():
    from qsar_tl.features.molecules import AutoQSARDescriptorSelector

    rng = np.random.RandomState(0)
    base = rng.normal(size=(60, 3))
    X = np.hstack([base, base * 2.0 + 1e-6, np.ones((60, 2))])  # copies + constants
    sel = AutoQSARDescriptorSelector(corr_threshold=0.8).fit(
        X, [f"d{i}" for i in range(X.shape[1])]
    )
    assert len(sel.selected_names) <= 4
    assert sel.transform(X).shape[1] == len(sel.selected_names)


# ------------------------------------------------------------------------------- reactions


def test_every_default_reaction_passes_its_self_test():
    """The reason the self-tests exist. A wrong SMARTS yields nothing, silently.

    The click template is the one this caught: without explicit +0/-0 on the product the
    azide's formal charges carry through, the ring valences break, sanitisation fails and
    the reaction produces nothing at all.
    """
    from qsar_tl.screen.enumerate import DEFAULT_REACTIONS, validate_reactions

    library = validate_reactions(DEFAULT_REACTIONS, verbose=False)
    assert not library.failures, library.failures
    assert len(library) == len(DEFAULT_REACTIONS)


def test_a_deliberately_broken_reaction_is_disabled_not_trusted():
    from qsar_tl.screen.enumerate import Reaction, validate_reactions

    broken = Reaction(
        name="broken",
        smarts="[C:1](=[O:2])[OX2H1].[NX3:3]>>[C:1](=[O:2])[N:3]",
        roles=("acid", "amine"),
        test=(("CC(=O)O", "CN"), "CCCCCCCC"),  # not the real product
    )
    library = validate_reactions([broken], verbose=False)
    assert "broken" in library.failures
    assert len(library) == 0


def test_enumeration_produces_labelled_products():
    from qsar_tl.screen.enumerate import (
        BuildingBlock,
        DEFAULT_REACTIONS,
        enumerate_library,
        validate_reactions,
    )

    library = validate_reactions(DEFAULT_REACTIONS, verbose=False)
    cores = [BuildingBlock("amine1", "NCCCC(N)C(=O)O", "amine")]
    caps = [
        BuildingBlock("benzoyl", "OC(=O)c1ccccc1", "acid"),
        BuildingBlock("acetyl", "CC(=O)O", "acid"),
    ]
    frame = enumerate_library(cores, caps, library, verbose=False)
    assert len(frame) >= 2
    for col in ("smiles", "core_id", "reaction", "cap_id", "cap_role", "compound_id"):
        assert col in frame.columns
    assert set(frame["cap_id"]) == {"benzoyl", "acetyl"}


def test_unparseable_building_blocks_are_skipped_not_fatal():
    from qsar_tl.screen.enumerate import (
        BuildingBlock,
        DEFAULT_REACTIONS,
        enumerate_library,
        validate_reactions,
    )

    library = validate_reactions(DEFAULT_REACTIONS, verbose=False)
    frame = enumerate_library(
        [BuildingBlock("bad", "@@@nonsense@@@", "amine"),
         BuildingBlock("good", "NCCCC(N)C(=O)O", "amine")],
        [BuildingBlock("acetyl", "CC(=O)O", "acid")],
        library, verbose=False,
    )
    assert set(frame["core_id"]) == {"good"}


def test_reaction_with_no_matching_block_yields_nothing_quietly():
    from qsar_tl.screen.enumerate import (
        BuildingBlock,
        DEFAULT_REACTIONS,
        enumerate_library,
        validate_reactions,
    )

    library = validate_reactions(DEFAULT_REACTIONS, verbose=False)
    # only an acid cap, so only amide_coupling can fire
    frame = enumerate_library(
        [BuildingBlock("amine1", "NCCCC(N)C(=O)O", "amine")],
        [BuildingBlock("acetyl", "CC(=O)O", "acid")],
        library, verbose=False,
    )
    assert set(frame["reaction"]) == {"amide_coupling"}


def test_property_window_comes_from_the_actives():
    from qsar_tl.screen.enumerate import property_window

    actives = ["C" * n + "C(=O)O" for n in range(5, 30)]
    window = property_window(actives, quantile=0.05, min_actives=5)
    assert window is not None
    assert set(window.index) == {"MolWt", "TPSA", "NumHDonors", "NumRotatableBonds"}
    assert (window["low"] <= window["high"]).all()


def test_property_window_needs_enough_actives():
    from qsar_tl.screen.enumerate import property_window

    assert property_window(["CCO"], min_actives=10) is None


def test_property_filter_floor_flags_instead_of_emptying():
    """Applied without a floor, a window from the actives once emptied a library."""
    from qsar_tl.screen.enumerate import apply_property_window

    frame = pd.DataFrame(
        {
            "smiles": ["C"] * 10,
            "MolWt": np.linspace(900, 1200, 10),      # all far above the window
            "TPSA": np.linspace(300, 400, 10),
            "NumHDonors": np.full(10, 9.0),
            "NumRotatableBonds": np.full(10, 20.0),
        }
    )
    window = pd.DataFrame(
        {"low": [100, 50, 0, 0], "high": [400, 150, 6, 10], "median": [250, 90, 3, 5]},
        index=["MolWt", "TPSA", "NumHDonors", "NumRotatableBonds"],
    )
    out = apply_property_window(frame, window, apply=True, min_keep_frac=0.2,
                                verbose=False)
    assert len(out) == 10                       # nothing removed
    assert "in_property_window" in out.columns
    assert not out["in_property_window"].any()  # but all flagged as outside


def test_property_filter_applies_when_most_products_survive():
    from qsar_tl.screen.enumerate import apply_property_window

    frame = pd.DataFrame(
        {
            "smiles": ["C"] * 10,
            "MolWt": np.linspace(200, 380, 10),
            "TPSA": np.full(10, 100.0),
            "NumHDonors": np.full(10, 3.0),
            "NumRotatableBonds": np.full(10, 5.0),
        }
    )
    frame.loc[0, "MolWt"] = 2000.0  # one outlier
    window = pd.DataFrame(
        {"low": [100, 50, 0, 0], "high": [400, 150, 6, 10], "median": [250, 90, 3, 5]},
        index=["MolWt", "TPSA", "NumHDonors", "NumRotatableBonds"],
    )
    out = apply_property_window(frame, window, apply=True, min_keep_frac=0.2,
                                verbose=False)
    assert len(out) == 9


def test_positive_control_has_two_tiers():
    from qsar_tl.features.molecules import inchikey
    from qsar_tl.screen.enumerate import positive_control
    from rdkit import Chem

    one = inchikey(Chem.MolFromSmiles("N[C@@H](CCC(=O)O)C(=O)O"))
    other = inchikey(Chem.MolFromSmiles("N[C@H](CCC(=O)O)C(=O)O"))
    result = positive_control([other], training_keys={one})
    assert result["n_exact"] == 0          # the full keys differ
    assert result["n_connectivity"] == 1   # the constitution is the same


# ------------------------------------------------------------------------- library loading


def test_resolve_missing_library_raises_and_does_not_substitute(tmp_path):
    """A silent demo-file fallback once made a run screen 8 molecules."""
    from qsar_tl.screen.library import resolve_library_path

    with pytest.raises(FileNotFoundError, match="does not substitute"):
        resolve_library_path(str(tmp_path / "nope.sdf"))


def test_absolute_path_is_not_rewritten_when_it_exists(tmp_path):
    from qsar_tl.screen.library import resolve_library_path

    path = tmp_path / "lib.csv"
    path.write_text("smiles\nCCO\n")
    assert resolve_library_path(str(path)) == path


def test_csv_library_columns_are_sniffed(tmp_path):
    from qsar_tl.screen.library import load_library

    path = tmp_path / "lib.csv"
    path.write_text("idnumber,SMILES\nZ1,CCO\nZ2,c1ccccc1\n")
    frame = load_library(str(path), verbose=False)
    assert list(frame.columns) == ["smiles", "compound_id"]
    assert list(frame["compound_id"]) == ["Z1", "Z2"]


def test_smi_library(tmp_path):
    from qsar_tl.screen.library import load_library

    path = tmp_path / "lib.smi"
    path.write_text("CCO ethanol\nc1ccccc1 benzene\n")
    frame = load_library(str(path), verbose=False)
    assert list(frame["compound_id"]) == ["ethanol", "benzene"]


def test_library_health_report_flags_a_fragment_collection(tmp_path):
    from qsar_tl.screen.library import library_health_report

    frame = pd.DataFrame({"smiles": ["C", "CC", "CCO", "CO"] * 10})
    health = library_health_report(frame, n_sample=100)
    assert health.fragment_like_frac > 0.5
    assert health.looks_druglike is False
    assert "fragment" in health.summary()


# --------------------------------------------------------------------- applicability domain


def test_applicability_domain_flags_distant_chemistry():
    from qsar_tl.domain.applicability import ApplicabilityDomain
    from qsar_tl.features.molecules import compute_descriptors, featurise

    train = pd.DataFrame({"smiles": ["C" * n + "C(=O)O" for n in range(4, 24)]})
    fs = featurise(train, n_bits=256, with_descriptors=True, progress=False)
    ad = ApplicabilityDomain(latent_pca_components=4, unfamiliarity_k=3).fit(
        fs.fingerprints, fs.descriptors
    )
    assert set(ad.thresholds) == {
        "tanimoto", "descriptor_manhattan", "latent_manhattan", "unfamiliarity"
    }

    # something structurally unrelated to a homologous series of fatty acids
    far = pd.DataFrame({"smiles": ["c1ccc2c(c1)nc(n2)N3CCN(CC3)c4ccccn4"]})
    far_fs = featurise(far, n_bits=256, with_descriptors=True, progress=False)
    far_desc, _ = compute_descriptors(
        [__import__("rdkit").Chem.MolFromSmiles(far["smiles"].iloc[0])],
        names=fs.descriptor_names,
    )
    scores = ad.score(far_fs.fingerprints, far_desc)
    assert bool(scores["tanimoto_OOD"].iloc[0])
    assert bool(scores["consensus_OOD"].iloc[0])   # the veto alone is enough


def test_tanimoto_veto_is_stricter_than_consensus():
    from qsar_tl.domain.applicability import ApplicabilityDomain
    from qsar_tl.features.molecules import featurise

    train = pd.DataFrame({"smiles": ["C" * n + "C(=O)O" for n in range(4, 24)]})
    fs = featurise(train, n_bits=256, with_descriptors=True, progress=False)
    kwargs = dict(latent_pca_components=4, unfamiliarity_k=3)
    with_veto = ApplicabilityDomain(tanimoto_veto=True, **kwargs).fit(
        fs.fingerprints, fs.descriptors
    )
    without = ApplicabilityDomain(tanimoto_veto=False, **kwargs).fit(
        fs.fingerprints, fs.descriptors
    )
    a = with_veto.score(fs.fingerprints, fs.descriptors)
    b = without.score(fs.fingerprints, fs.descriptors)
    assert int(a["consensus_OOD"].sum()) >= int(b["consensus_OOD"].sum())


def test_butina_clustering_groups_a_homologous_series():
    from qsar_tl.evaluation.splits import butina_clusters_from_fingerprints
    from qsar_tl.features.molecules import featurise

    frame = pd.DataFrame(
        {"smiles": ["C" * n + "C(=O)O" for n in range(4, 14)]
         + ["c1ccccc1" + "C" * n for n in range(1, 11)]}
    )
    fs = featurise(frame, n_bits=512, progress=False)
    groups = butina_clusters_from_fingerprints(fs.fingerprints, 0.4, progress=False)
    assert len(np.unique(groups)) >= 2
    assert len(groups) == len(fs)
