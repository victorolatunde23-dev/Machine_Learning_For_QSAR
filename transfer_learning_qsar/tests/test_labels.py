"""Derived-value detection, censored recovery and label assembly.

These cover the single most consequential piece of data cleaning in the package: on the
real dataset the derived-value detector removes 141 of 169 dual-measured pKi values as
Cheng-Prusoff transforms of the IC50.
"""

import numpy as np
import pandas as pd
import pytest

from qsar_tl.data.labels import (
    assemble_label_table,
    build_endpoint_labels,
    detect_derived_endpoint,
    source_tag,
)


# ------------------------------------------------------------------------------ provenance


@pytest.mark.parametrize(
    "name, expected",
    [
        ("US20250213705, Compound 50", "US20250213705"),
        ("US 10736974, Example 3", "US10736974"),
        ("WO2015055318 ligand", "WO2015055318"),
        ("EP1234567", "EP1234567"),
        ("CHEMBL1921897", "CHEMBL"),
        ("chembl65860::some-name", "CHEMBL"),
        ("PSMA-617", "other"),
        ("", "unknown"),
        (None, "unknown"),
        (float("nan"), "unknown"),
    ],
)
def test_source_tag(name, expected):
    assert source_tag(name) == expected


def test_patent_wins_over_chembl_when_both_present():
    """A filing is a real source; CHEMBL is a catch-all bucket. The filing must win,
    because the evaluation holds a dominant filing out as its own fold."""
    assert source_tag("US20250213705 (also CHEMBL123)") == "US20250213705"


# -------------------------------------------------------------------- derived-value blocks


def test_detects_both_fixed_offset_blocks(wide_frame):
    report = detect_derived_endpoint(wide_frame, "pIC50", "pKi")
    ratios = sorted(round(b.implied_ratio) for b in report.blocks)
    assert 140 in ratios
    assert 5 in ratios


def test_flags_the_derived_compounds_and_not_the_independent_ones(wide_frame):
    report = detect_derived_endpoint(wide_frame, "pIC50", "pKi")
    flagged = set(wide_frame.loc[report.mask, "ligand_name"])
    assert any("US20250213705" in n for n in flagged)
    assert any("US10736974" in n for n in flagged)
    # the independent block has random scatter and must survive
    independent = {
        n for n in wide_frame["ligand_name"] if str(n).startswith("CHEMBL1")
    }
    assert not (flagged & independent)


def test_each_compound_belongs_to_at_most_one_block(wide_frame):
    report = detect_derived_endpoint(wide_frame, "pIC50", "pKi")
    assert sum(b.n_compounds for b in report.blocks) == report.n_flagged


def test_rounding_buckets_are_merged_into_one_block():
    """One block must not be reported three times under adjacent rounding buckets.

    An earlier version printed a single block of 111 compounds as three rows of
    106/103/102 because the tolerance windows around 2.14 / 2.15 / 2.16 overlapped.
    """
    rng = np.random.RandomState(0)
    n = 90
    pic = rng.uniform(5, 9, n)
    # a single block whose offsets straddle three rounding buckets
    offsets = rng.choice([2.144, 2.150, 2.156], size=n)
    wide = pd.DataFrame(
        {
            "smiles": [f"C{i}" for i in range(n)],
            "pIC50": pic,
            "pKi": pic + offsets,
            "source_tag": ["US1234567"] * n,
        }
    )
    report = detect_derived_endpoint(wide, "pIC50", "pKi", min_block_size=5,
                                     offset_tol=0.03)
    assert len(report.blocks) == 1, [b.as_row() for b in report.blocks]
    assert report.blocks[0].n_compounds >= 80


def test_small_tail_clumps_are_rejected_at_the_default_block_size(wide_frame):
    """A handful of compounds landing in one window by chance is not a conversion.

    Lowering min_block_size to 5 lets tail clumps through -- the enrichment guard's
    expectation falls below one compound out in the tails, so the bar drops to the
    absolute floor. The default of 10 is what keeps that shut, and this test is here so
    that anyone tempted to lower it sees the trade.
    """
    permissive = detect_derived_endpoint(wide_frame, "pIC50", "pKi", min_block_size=5)
    default = detect_derived_endpoint(wide_frame, "pIC50", "pKi")
    assert default.n_flagged <= permissive.n_flagged
    independent = {
        n for n in wide_frame["ligand_name"] if str(n).startswith("CHEMBL1")
    }
    assert not (set(wide_frame.loc[default.mask, "ligand_name"]) & independent)


def test_enrichment_guard_can_be_disabled():
    """min_enrichment=0 restores the older, permissive behaviour, for comparison."""
    rng = np.random.RandomState(7)
    pic = rng.uniform(5, 9, 200)
    wide = pd.DataFrame(
        {"smiles": [f"C{i}" for i in range(200)], "pIC50": pic,
         "pKi": pic + rng.normal(0, 0.8, 200), "source_tag": ["other"] * 200}
    )
    strict = detect_derived_endpoint(wide, "pIC50", "pKi", min_block_size=10)
    loose = detect_derived_endpoint(wide, "pIC50", "pKi", min_block_size=10,
                                    min_enrichment=0.0)
    assert strict.n_flagged == 0
    assert loose.n_flagged >= strict.n_flagged


def test_whole_dataset_converted_is_still_detected():
    """When every dual-measured compound was converted there is no background scatter.

    The enrichment test is meaningless then -- a Gaussian fitted to the block predicts the
    block -- so it is skipped once the remaining offsets are tighter than a few window
    widths. Without that escape a fully converted dataset reports no blocks at all.
    """
    rng = np.random.RandomState(3)
    n = 120
    pic = rng.uniform(5, 9, n)
    wide = pd.DataFrame(
        {"smiles": [f"C{i}" for i in range(n)], "pIC50": pic,
         "pKi": pic + np.log10(140.0) + rng.normal(0, 0.004, n),
         "source_tag": ["US1234567"] * n}
    )
    report = detect_derived_endpoint(wide, "pIC50", "pKi")
    assert len(report.blocks) == 1
    assert round(report.blocks[0].implied_ratio) == 140
    assert report.fraction_flagged > 0.9


def test_no_blocks_when_endpoints_are_independent():
    rng = np.random.RandomState(1)
    pic = rng.uniform(5, 9, 120)
    wide = pd.DataFrame(
        {"smiles": [f"C{i}" for i in range(120)], "pIC50": pic,
         "pKi": pic + rng.normal(0, 0.8, 120), "source_tag": ["other"] * 120}
    )
    report = detect_derived_endpoint(wide, "pIC50", "pKi", min_block_size=5)
    assert report.blocks == []
    assert report.n_flagged == 0
    assert "independently measured" in report.summary()


def test_missing_columns_return_an_empty_report():
    wide = pd.DataFrame({"smiles": ["C"], "pIC50": [7.0]})
    report = detect_derived_endpoint(wide, "pIC50", "pKi")
    assert report.n_flagged == 0
    assert report.to_frame().empty


def test_report_frame_and_fraction(wide_frame):
    report = detect_derived_endpoint(wide_frame, "pIC50", "pKi")
    frame = report.to_frame()
    assert list(frame.columns)[:3] == ["offset", "implied_ratio", "n_compounds"]
    assert 0.0 < report.fraction_flagged <= 1.0
    assert "Cheng-Prusoff" in report.summary()


# ------------------------------------------------------------------------ censored labels


def test_censored_upper_bound_below_cut_becomes_a_negative():
    long = pd.DataFrame(
        [{"smiles": "CCO", "ligand_name": "x", "affinity_raw": ">100000",
          "endpoint_name": "pKi"}]
    )
    records, stats = build_endpoint_labels(long, "pKi", 8.0)
    assert records["CCO"]["y"] == 0
    assert records["CCO"]["label_source"] == "censored"
    assert stats.recovered_inactive == 1


def test_censored_lower_bound_above_cut_becomes_a_positive():
    long = pd.DataFrame(
        [{"smiles": "CCO", "ligand_name": "x", "affinity_raw": "<0.5",
          "endpoint_name": "pKi"}]
    )
    records, stats = build_endpoint_labels(long, "pKi", 8.0)
    assert records["CCO"]["y"] == 1
    assert stats.recovered_active == 1


def test_bound_straddling_the_cut_is_dropped_as_ambiguous():
    long = pd.DataFrame(
        [{"smiles": "CCO", "ligand_name": "x", "affinity_raw": ">0.5",
          "endpoint_name": "pKi"}]
    )
    records, stats = build_endpoint_labels(long, "pKi", 8.0)
    assert records == {}
    assert stats.dropped_ambiguous == 1


def test_censored_can_be_switched_off():
    long = pd.DataFrame(
        [{"smiles": "CCO", "ligand_name": "x", "affinity_raw": ">100000",
          "endpoint_name": "pKi"}]
    )
    records, stats = build_endpoint_labels(long, "pKi", 8.0, keep_censored=False)
    assert records == {}
    assert stats.recovered_inactive == 0


def test_most_informative_gt_bound_is_the_smallest():
    """Several '>' bounds must all hold, so the smallest carries the most information."""
    long = pd.DataFrame(
        [
            {"smiles": "CCO", "ligand_name": "x", "affinity_raw": ">100000",
             "endpoint_name": "pKi"},
            {"smiles": "CCO", "ligand_name": "x", "affinity_raw": ">1000",
             "endpoint_name": "pKi"},
        ]
    )
    records, _ = build_endpoint_labels(long, "pKi", 8.0)
    # p_upper = min over bounds = 9 - log10(100000) = 4, which is <= 8 -> negative
    assert records["CCO"]["y"] == 0


def test_a_numeric_value_takes_precedence_over_a_bound():
    long = pd.DataFrame(
        [
            {"smiles": "CCO", "ligand_name": "x", "affinity_raw": "5",
             "endpoint_name": "pKi"},
            {"smiles": "CCO", "ligand_name": "x", "affinity_raw": ">100000",
             "endpoint_name": "pKi"},
        ]
    )
    records, _ = build_endpoint_labels(long, "pKi", 8.0)
    assert records["CCO"]["label_source"] == "numeric"
    assert records["CCO"]["y"] == 1  # 5 nM -> pKi 8.3


def test_replicates_are_aggregated_by_median():
    long = pd.DataFrame(
        [
            {"smiles": "CCO", "ligand_name": "x", "affinity_raw": a,
             "endpoint_name": "pKi"}
            for a in ("1", "10", "100")
        ]
    )
    records, stats = build_endpoint_labels(long, "pKi", 8.0)
    assert records["CCO"]["p_affinity"] == pytest.approx(8.0)
    assert records["CCO"]["n_measurements"] == 3
    assert stats.numeric_rows == 3


def test_values_outside_the_affinity_window_are_discarded():
    long = pd.DataFrame(
        [
            {"smiles": "A", "ligand_name": "a", "affinity_raw": "1e12",
             "endpoint_name": "pKi"},
            {"smiles": "B", "ligand_name": "b", "affinity_raw": "1e-9",
             "endpoint_name": "pKi"},
        ]
    )
    records, _ = build_endpoint_labels(long, "pKi", 8.0, min_affinity_nm=1e-3,
                                       max_affinity_nm=1e6)
    assert records == {}


def test_derived_smiles_are_dropped_for_that_endpoint_only():
    long = pd.DataFrame(
        [
            {"smiles": "CCO", "ligand_name": "x", "affinity_raw": "5",
             "endpoint_name": "pKi"},
            {"smiles": "CCO", "ligand_name": "x", "affinity_raw": "700",
             "endpoint_name": "pIC50"},
        ]
    )
    recs_ki, stats_ki = build_endpoint_labels(long, "pKi", 8.0,
                                              derived_smiles={"CCO"})
    recs_ic, _ = build_endpoint_labels(long, "pIC50", 8.0, derived_smiles=set())
    assert recs_ki == {}
    assert stats_ki.dropped_derived == 1
    assert "CCO" in recs_ic  # the measured endpoint survives


# ----------------------------------------------------------------------- assembled table


def test_assembly_keys_derived_removal_off_derived_endpoint_not_primary(long_frame,
                                                                       wide_frame):
    """The bug, pinned. With pKi primary AND derived, keying the test off the primary
    endpoint kept every Cheng-Prusoff value. The two fields are separate for this reason.
    """
    report = detect_derived_endpoint(wide_frame, "pIC50", "pKi")
    derived = set(wide_frame.loc[report.mask, "smiles"])
    assert derived, "fixture should contain derived compounds"

    table, stats = assemble_label_table(
        long_frame,
        endpoints=["pKi"],
        active_thresholds={"pKi": 8.0},
        derived_endpoint="pKi",
        drop_derived=True,
        derived_smiles=derived,
    )
    assert stats["pKi"].dropped_derived > 0
    kept = set(table.loc[table["y_pKi"].notna(), "smiles"])
    assert not (kept & derived)


def test_assembly_with_derived_removal_off_keeps_them(long_frame, wide_frame):
    report = detect_derived_endpoint(wide_frame, "pIC50", "pKi")
    derived = set(wide_frame.loc[report.mask, "smiles"])
    table, stats = assemble_label_table(
        long_frame,
        endpoints=["pKi"],
        active_thresholds={"pKi": 8.0},
        derived_endpoint="pKi",
        drop_derived=False,
        derived_smiles=derived,
    )
    assert stats["pKi"].dropped_derived == 0
    kept = set(table.loc[table["y_pKi"].notna(), "smiles"])
    assert kept & derived


def test_assembly_produces_expected_columns(long_frame):
    table, _ = assemble_label_table(
        long_frame,
        endpoints=["pIC50", "pKi"],
        active_thresholds={"pIC50": 8.0, "pKi": 8.0},
        derived_endpoint=None,
        drop_derived=False,
    )
    for col in ("smiles", "ligand_name", "source_tag", "n_labels",
                "pIC50", "y_pIC50", "src_pIC50", "pKi", "y_pKi", "src_pKi"):
        assert col in table.columns
    assert table["n_labels"].max() == 2


def test_extra_smiles_are_carried_with_nan_labels(long_frame):
    """Compounds measured only on an unmodelled endpoint still define the domain."""
    extra = ["TOTALLY_NEW_SMILES"]
    table, _ = assemble_label_table(
        long_frame,
        endpoints=["pKi"],
        active_thresholds={"pKi": 8.0},
        derived_endpoint=None,
        drop_derived=False,
        extra_smiles=extra,
    )
    row = table[table["smiles"] == "TOTALLY_NEW_SMILES"]
    assert len(row) == 1
    assert pd.isna(row.iloc[0]["y_pKi"])


def test_empty_input_returns_an_empty_table_with_columns():
    empty = pd.DataFrame(
        columns=["smiles", "ligand_name", "affinity_raw", "endpoint_name"]
    )
    table, _ = assemble_label_table(
        empty, endpoints=["pKi"], active_thresholds={"pKi": 8.0},
        derived_endpoint=None, drop_derived=False,
    )
    assert table.empty
    assert "y_pKi" in table.columns
