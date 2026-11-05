"""The streaming scorer, driven with an injected featuriser so no toolkit is needed.

The featuriser and scorer are parameters of :func:`stream_and_score` precisely so that the
chunking, deduplication, failure counting, threshold application and consensus logic can
be tested for real rather than mocked around.
"""

import numpy as np
import pandas as pd
import pytest

from qsar_tl.screen.stream import (
    consensus_tiers,
    iter_smiles_records,
    stream_and_score,
)


def fake_featuriser(smiles_list):
    """`smiles` -> an 8-dim vector. Anything containing 'X' fails to parse."""
    kept, rows, carriers = [], [], []
    for i, smi in enumerate(smiles_list):
        if "X" in smi or not smi:
            continue
        vec = np.zeros(8, dtype=np.float32)
        for j, ch in enumerate(smi[:8]):
            vec[j] = (ord(ch) % 7) / 7.0
        kept.append(i)
        rows.append(vec)
        carriers.append((smi, f"KEY-{smi.strip('0123456789')}"))
    X = np.vstack(rows) if rows else np.zeros((0, 8), dtype=np.float32)
    return X, kept, carriers


def fake_scorer(X, carriers):
    """Two 'models' with different but correlated opinions."""
    base = X.mean(axis=1) if len(X) else np.zeros(0)
    return {"A": np.clip(base, 0, 1), "B": np.clip(base * 0.8 + 0.1, 0, 1)}


def records(n=50, bad_every=0, dup_every=0):
    out = []
    for i in range(n):
        if bad_every and i % bad_every == 0:
            out.append((f"id{i}", "XXX"))
            continue
        if dup_every and i % dup_every == 0 and i > 0:
            out.append((f"id{i}", "CCCO"))  # same key every time
            continue
        out.append((f"id{i}", "C" * (2 + i % 9) + "O" + str(i)))
    return out


def test_scores_every_record_and_adds_expected_columns():
    result = stream_and_score(
        records(30), featurise_chunk=fake_featuriser, score_chunk=fake_scorer,
        thresholds={"A": 0.5, "B": 0.5}, chunk_size=7, progress=False,
    )
    assert result.n_read == 30
    assert result.n_scored == 30
    frame = result.frame
    for col in ("compound_id", "smiles", "p_A", "p_B", "call_A", "call_B",
                "n_models_active", "p_mean", "in_training"):
        assert col in frame.columns


def test_chunk_size_does_not_change_the_result():
    kwargs = dict(
        featurise_chunk=fake_featuriser, score_chunk=fake_scorer,
        thresholds={"A": 0.5, "B": 0.5}, progress=False,
    )
    small = stream_and_score(records(40), chunk_size=3, **kwargs).frame
    large = stream_and_score(records(40), chunk_size=1000, **kwargs).frame
    pd.testing.assert_frame_equal(small, large)


def test_identifiers_stay_aligned_when_a_chunk_loses_rows():
    """The failure that chunked streaming invites: ids drifting off their molecules."""
    result = stream_and_score(
        records(40, bad_every=5), featurise_chunk=fake_featuriser,
        score_chunk=fake_scorer, thresholds={"A": 0.5}, chunk_size=6, progress=False,
    )
    frame = result.frame
    assert result.n_failed == 8
    assert result.n_scored == 32
    # every surviving row's id must still match its own smiles
    for ident, smi in zip(frame["compound_id"], frame["smiles"]):
        assert smi.endswith(ident.replace("id", ""))


def test_duplicates_are_counted_and_dropped():
    result = stream_and_score(
        records(30, dup_every=5), featurise_chunk=fake_featuriser,
        score_chunk=fake_scorer, thresholds={"A": 0.5}, chunk_size=4,
        key_fn=lambda carrier: carrier[1], progress=False,
    )
    assert result.n_duplicates > 0
    assert result.n_scored + result.n_duplicates + result.n_failed == result.n_read
    assert result.frame["inchikey"].is_unique


def test_dedup_is_off_without_a_key_function():
    result = stream_and_score(
        records(30, dup_every=5), featurise_chunk=fake_featuriser,
        score_chunk=fake_scorer, thresholds={"A": 0.5}, chunk_size=4, progress=False,
    )
    assert result.n_duplicates == 0


def test_dedup_spans_chunk_boundaries():
    """A key seen in chunk 1 must still be recognised in chunk 9."""
    recs = [("a", "CCCO"), ("b", "CCCCO")] + [("c", "CCCO")] * 5
    result = stream_and_score(
        recs, featurise_chunk=fake_featuriser, score_chunk=fake_scorer,
        thresholds={"A": 0.5}, chunk_size=1,
        key_fn=lambda carrier: carrier[1], progress=False,
    )
    assert result.n_scored == 2
    assert result.n_duplicates == 5


def test_training_compounds_are_flagged_not_dropped():
    """They are a free positive control: where they rank is the first thing to check."""
    result = stream_and_score(
        [("a", "CCCO"), ("b", "CCCCCO")], featurise_chunk=fake_featuriser,
        score_chunk=fake_scorer, thresholds={"A": 0.5}, chunk_size=10,
        key_fn=lambda carrier: carrier[1], training_keys={"KEY-CCCO"}, progress=False,
    )
    assert result.n_scored == 2
    assert result.frame["in_training"].sum() == 1


def test_thresholds_drive_the_calls():
    low = stream_and_score(
        records(30), featurise_chunk=fake_featuriser, score_chunk=fake_scorer,
        thresholds={"A": 0.01}, chunk_size=10, progress=False,
    ).frame
    high = stream_and_score(
        records(30), featurise_chunk=fake_featuriser, score_chunk=fake_scorer,
        thresholds={"A": 0.99}, chunk_size=10, progress=False,
    ).frame
    assert low["call_A"].sum() > high["call_A"].sum()


def test_per_model_thresholds_are_independent():
    frame = stream_and_score(
        records(30), featurise_chunk=fake_featuriser, score_chunk=fake_scorer,
        thresholds={"A": 0.01, "B": 0.99}, chunk_size=10, progress=False,
    ).frame
    assert frame["call_A"].sum() > frame["call_B"].sum()
    assert set(frame["n_models_active"].unique()) <= {0, 1, 2}


def test_output_csv_matches_the_returned_frame(tmp_path):
    out = tmp_path / "preds.csv"
    result = stream_and_score(
        records(25), featurise_chunk=fake_featuriser, score_chunk=fake_scorer,
        thresholds={"A": 0.5, "B": 0.5}, chunk_size=4, output_csv=out, progress=False,
    )
    on_disk = pd.read_csv(out)
    assert len(on_disk) == result.n_scored
    assert list(on_disk.columns) == list(result.frame.columns)


def test_csv_has_one_header_despite_many_chunks(tmp_path):
    out = tmp_path / "preds.csv"
    stream_and_score(
        records(25), featurise_chunk=fake_featuriser, score_chunk=fake_scorer,
        thresholds={"A": 0.5}, chunk_size=3, output_csv=out, progress=False,
    )
    text = out.read_text()
    assert text.count("compound_id") == 1


def test_existing_output_is_replaced_not_appended(tmp_path):
    out = tmp_path / "preds.csv"
    out.write_text("stale,data\n1,2\n")
    result = stream_and_score(
        records(10), featurise_chunk=fake_featuriser, score_chunk=fake_scorer,
        thresholds={"A": 0.5}, chunk_size=5, output_csv=out, progress=False,
    )
    assert "stale" not in out.read_text()
    assert len(pd.read_csv(out)) == result.n_scored


def test_limit_stops_early():
    result = stream_and_score(
        records(100), featurise_chunk=fake_featuriser, score_chunk=fake_scorer,
        thresholds={"A": 0.5}, chunk_size=7, limit=20, progress=False,
    )
    assert result.n_read == 20


def test_return_frame_can_be_disabled_for_memory(tmp_path):
    out = tmp_path / "preds.csv"
    result = stream_and_score(
        records(20), featurise_chunk=fake_featuriser, score_chunk=fake_scorer,
        thresholds={"A": 0.5}, chunk_size=5, output_csv=out, progress=False,
        return_frame=False,
    )
    assert result.frame is None
    assert len(pd.read_csv(out)) == 20


def test_all_records_failing_is_not_an_error():
    result = stream_and_score(
        [("a", "XXX"), ("b", "XXX")], featurise_chunk=fake_featuriser,
        score_chunk=fake_scorer, thresholds={"A": 0.5}, chunk_size=5, progress=False,
    )
    assert result.n_scored == 0
    assert result.n_failed == 2
    assert result.frame.empty


def test_empty_input():
    result = stream_and_score(
        [], featurise_chunk=fake_featuriser, score_chunk=fake_scorer,
        thresholds={"A": 0.5}, progress=False,
    )
    assert result.n_read == 0 and result.n_scored == 0


def test_summary_mentions_the_counters():
    result = stream_and_score(
        records(20, bad_every=7), featurise_chunk=fake_featuriser,
        score_chunk=fake_scorer, thresholds={"A": 0.5}, chunk_size=5, progress=False,
    )
    text = result.summary()
    assert "records read" in text and "unique compounds scored" in text


def test_consensus_tiers_counts_every_tier():
    frame = pd.DataFrame({"n_models_active": [2, 2, 1, 0, 0, 0]})
    tiers = consensus_tiers(frame, 2)
    assert tiers[2] == 2 and tiers[1] == 1 and tiers[0] == 3


def test_iter_smiles_records_uses_the_id_column():
    frame = pd.DataFrame({"smiles": ["CC", "CCC"], "compound_id": ["x", "y"]})
    assert list(iter_smiles_records(frame)) == [("x", "CC"), ("y", "CCC")]


def test_iter_smiles_records_without_an_id_column():
    frame = pd.DataFrame({"smiles": ["CC", "CCC"]})
    assert [i for i, _ in iter_smiles_records(frame)] == ["row_0", "row_1"]
