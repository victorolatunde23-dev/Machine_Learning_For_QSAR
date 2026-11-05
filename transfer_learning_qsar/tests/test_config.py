"""Config loading and validation.

The validation exists because a real run was lost to a config field that was read but
never defined: the download silently fell back to synthetic data and the failure surfaced
six stages later as an unrelated sklearn error. These tests pin that class of bug shut.
"""

import pytest
import yaml

from qsar_tl.config import Config, ConfigError, load_config


def test_defaults_validate():
    Config().validate()


def test_round_trip_through_dict():
    cfg = Config()
    again = Config.from_dict(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()


def test_yaml_round_trip(tmp_path):
    cfg = Config(run_name="rt")
    cfg.evaluation.butina_cutoff = 0.35
    path = cfg.to_yaml(tmp_path / "c.yaml")
    back = Config.from_yaml(path)
    assert back.run_name == "rt"
    assert back.evaluation.butina_cutoff == 0.35


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown config key"):
        Config.from_dict({"run_nmae": "typo"})


def test_unknown_nested_key_is_rejected_and_names_the_section():
    with pytest.raises(ConfigError) as exc:
        Config.from_dict({"evaluation": {"butina_cutof": 0.4}})
    assert "evaluation" in str(exc.value)


def test_unknown_key_suggests_the_intended_one():
    with pytest.raises(ConfigError) as exc:
        Config.from_dict({"features": {"feature_sett": "fp_only"}})
    assert "feature_set" in str(exc.value)


def test_non_strict_mode_ignores_unknown_keys():
    cfg = Config.from_dict({"run_nmae": "typo"}, strict=False)
    assert cfg.run_name == "run"


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"features": {"feature_set": "nonsense"}}, "feature_set"),
        ({"labels": {"task": "ranking"}}, "task"),
        ({"evaluation": {"calibration": "platt"}}, "calibration"),
        ({"evaluation": {"threshold_objective": "auc"}}, "threshold_objective"),
        ({"screen": {"source": "sdf"}}, "source"),
        ({"evaluation": {"threshold_max_called_frac": 0}}, "threshold_max_called_frac"),
        ({"evaluation": {"threshold_override": 1.5}}, "threshold_override"),
        ({"domain": {"consensus_min_flags": 9}}, "consensus_min_flags"),
        ({"screen": {"chunk_size": 0}}, "chunk_size"),
    ],
)
def test_bad_values_are_caught(patch, message):
    cfg = Config.from_dict(patch)
    with pytest.raises(ConfigError, match=message):
        cfg.validate()


def test_primary_endpoint_must_be_modelled():
    cfg = Config.from_dict(
        {"labels": {"modelled_endpoints": ["pKi"], "primary_endpoint": "pIC50"}}
    )
    with pytest.raises(ConfigError, match="primary_endpoint"):
        cfg.validate()


def test_modelled_endpoint_must_be_pulled():
    cfg = Config.from_dict(
        {
            "data": {"endpoint_names": ["pIC50"], "affinity_columns": ["IC50 (nM)"]},
            "labels": {"modelled_endpoints": ["pKd"], "primary_endpoint": "pKd",
                       "active_thresholds": {"pKd": 8.0}},
        }
    )
    with pytest.raises(ConfigError, match="not in"):
        cfg.validate()


def test_endpoint_and_column_lists_must_match_length():
    cfg = Config.from_dict(
        {"data": {"endpoint_names": ["pIC50", "pKi"], "affinity_columns": ["IC50 (nM)"]}}
    )
    with pytest.raises(ConfigError, match="differ in length"):
        cfg.validate()


def test_missing_active_threshold_is_caught():
    cfg = Config.from_dict(
        {"labels": {"modelled_endpoints": ["pKi"], "primary_endpoint": "pKi",
                    "active_thresholds": {"pIC50": 8.0}}}
    )
    with pytest.raises(ConfigError, match="active_thresholds"):
        cfg.validate()


def test_dropping_derived_values_needs_a_derived_endpoint():
    """The exact bug: with pKi primary, keying the test off primary kept every
    Cheng-Prusoff value. Making derived_endpoint a separate, required field is the fix."""
    cfg = Config.from_dict(
        {"labels": {"drop_derived_endpoint_values": True, "derived_endpoint": None}}
    )
    with pytest.raises(ConfigError, match="derived_endpoint"):
        cfg.validate()


def test_dropping_derived_values_needs_two_endpoints_pulled():
    cfg = Config.from_dict(
        {
            "data": {"endpoint_names": ["pKi"], "affinity_columns": ["Ki (nM)"]},
            "labels": {"modelled_endpoints": ["pKi"], "primary_endpoint": "pKi",
                       "drop_derived_endpoint_values": True, "derived_endpoint": "pKi"},
        }
    )
    with pytest.raises(ConfigError, match="two endpoints"):
        cfg.validate()


def test_unknown_model_is_caught_with_a_suggestion():
    cfg = Config.from_dict({"models": {"enabled": ["LightGMB"]}})
    with pytest.raises(ConfigError) as exc:
        cfg.validate()
    assert "LightGBM" in str(exc.value)


def test_committee_member_not_enabled_is_caught():
    """A committee naming a disabled model would silently drop it from the blend."""
    cfg = Config.from_dict(
        {
            "models": {
                "enabled": ["RandomForest"],
                "committee": {"pKi": ["RandomForest", "KPLS-DA"]},
                "floors": [],
            }
        }
    )
    with pytest.raises(ConfigError, match="KPLS-DA"):
        cfg.validate()


def test_floor_not_enabled_is_caught():
    """A floor has to be fitted to be compared against."""
    cfg = Config.from_dict(
        {"models": {"enabled": ["RandomForest"], "floors": ["kNN (Tanimoto)"]}}
    )
    with pytest.raises(ConfigError, match="floors"):
        cfg.validate()


def test_committee_for_unmodelled_endpoint_is_caught():
    cfg = Config.from_dict(
        {"models": {"enabled": ["RandomForest"], "floors": [],
                    "committee": {"pIC50": ["RandomForest"]}}}
    )
    with pytest.raises(ConfigError, match="not a modelled endpoint"):
        cfg.validate()


def test_file_source_needs_a_path():
    cfg = Config.from_dict({"screen": {"source": "file"}})
    with pytest.raises(ConfigError, match="library_path"):
        cfg.validate()


def test_mixed_source_needs_a_catalogue():
    cfg = Config.from_dict({"screen": {"source": "enumeration+library"}})
    with pytest.raises(ConfigError, match="mix_library_path"):
        cfg.validate()


def test_dotted_overrides_apply():
    cfg = load_config(None, overrides={"evaluation.butina_cutoff": 0.55,
                                       "run_name": "ov"})
    assert cfg.evaluation.butina_cutoff == 0.55
    assert cfg.run_name == "ov"


def test_overrides_are_validated():
    with pytest.raises(ConfigError):
        load_config(None, overrides={"features.feature_set": "bogus"})


def test_missing_file_raises():
    with pytest.raises(ConfigError, match="not found"):
        Config.from_yaml("/nonexistent/config.yaml")


def test_helpers():
    cfg = Config()
    assert cfg.endpoints == ["pKi"]
    assert cfg.is_multitask is False
    assert cfg.active_cut("pKi") == 8.0
    with pytest.raises(ConfigError, match="no active_threshold"):
        cfg.active_cut("pKd")
    # an empty committee falls back to every enabled model
    assert cfg.committee_for("pKi") == list(cfg.models.enabled)


def test_capability_report_mentions_unavailable_models():
    cfg = Config.from_dict({"models": {"enabled": ["RandomForest", "LightGBM"],
                                       "floors": []}})
    text = cfg.capability_report()
    assert "RandomForest" in text
