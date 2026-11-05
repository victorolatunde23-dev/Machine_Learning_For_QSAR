"""Command line interface.

::

    qsar-tl info
    qsar-tl validate configs/psma_pki_screen.yaml
    qsar-tl train    configs/psma_pki_screen.yaml
    qsar-tl screen   configs/psma_pki_screen.yaml --library Enamine.sdf
    qsar-tl enumerate --self-test

``validate`` is worth running first and costs nothing: it loads the config, rejects
unknown keys, and checks the cross-field combinations that would otherwise fail several
stages into a run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["main", "build_parser"]


def _parse_set(values: Optional[Sequence[str]]) -> Dict[str, Any]:
    """``--set a.b=1 --set c=x`` into ``{"a.b": 1, "c": "x"}``, with YAML-ish typing."""
    import yaml

    out: Dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            out[key.strip()] = yaml.safe_load(raw)
        except Exception:
            out[key.strip()] = raw
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qsar-tl",
        description="Configurable QSAR pipeline: labels, grouped evaluation, screening.",
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="what is installed and which models can run here")

    for name, help_text in (
        ("validate", "load a config, reject unknown keys, check consistency"),
        ("train", "labels -> features -> tuning -> grouped evaluation -> save"),
        ("screen", "score a compound library with the fitted models"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("config", nargs="?", help="path to a YAML config")
        p.add_argument(
            "--set", dest="overrides", action="append", metavar="KEY=VALUE",
            help="override a config value, e.g. --set evaluation.butina_cutoff=0.5",
        )
        p.add_argument("--output", help="override output_dir")
        p.add_argument("--quiet", action="store_true")
        if name == "screen":
            p.add_argument("--library", help="override screen.library_path")
            p.add_argument(
                "--limit", type=int, help="stop after N records (for a trial run)"
            )

    e = sub.add_parser("enumerate", help="reaction-based enumeration")
    e.add_argument(
        "--self-test", action="store_true",
        help="validate the reaction SMARTS and exit (needs rdkit)",
    )
    e.add_argument("config", nargs="?", help="path to a YAML config")
    e.add_argument("--output", help="where to write the enumerated library CSV")
    return parser


def _load(args) -> "Any":
    from .config import load_config

    overrides = _parse_set(getattr(args, "overrides", None))
    if getattr(args, "output", None):
        overrides["output_dir"] = args.output
    if getattr(args, "library", None):
        overrides["screen.library_path"] = args.library
        overrides.setdefault("screen.source", "file")
    if getattr(args, "limit", None):
        overrides["screen.limit"] = args.limit
    return load_config(args.config, overrides=overrides)


def _cmd_info() -> int:
    from . import capabilities

    print(capabilities())
    return 0


def _cmd_validate(args) -> int:
    from .config import ConfigError

    try:
        cfg = _load(args)
    except ConfigError as exc:
        print(f"INVALID\n{exc}", file=sys.stderr)
        return 2
    print("config OK\n")
    print(cfg.capability_report())
    return 0


def _cmd_train(args) -> int:
    from .pipeline import Pipeline

    cfg = _load(args)
    pipe = Pipeline(cfg, verbose=not args.quiet)
    pipe.run()
    print(f"\nartefacts in {cfg.output_dir}")
    return 0


def _cmd_screen(args) -> int:
    from .pipeline import Pipeline

    cfg = _load(args)
    if cfg.screen.source == "none":
        print(
            "screen.source is 'none'. Set it to 'file' (with screen.library_path) or "
            "pass --library.",
            file=sys.stderr,
        )
        return 2
    pipe = Pipeline(cfg, verbose=not args.quiet)
    pipe.run()
    from .screen.runner import screen_library

    result = screen_library(pipe)
    print(result.summary())
    return 0


def _cmd_enumerate(args) -> int:
    from ._optional import OptionalDependencyError
    from .screen.enumerate import DEFAULT_REACTIONS, validate_reactions

    try:
        library = validate_reactions(DEFAULT_REACTIONS, verbose=True)
    except OptionalDependencyError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.self_test:
        return 0 if not library.failures else 1
    if not args.config:
        print(
            "enumeration needs a config naming the building blocks; "
            "see configs/psma_pki_enumeration.yaml",
            file=sys.stderr,
        )
        return 2
    print(
        "Building blocks come from the config's screen section; see "
        "examples/enumerate_psma.py for a worked example."
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from . import __version__

        print(__version__)
        return 0
    if not args.command:
        parser.print_help()
        return 0
    if args.command == "info":
        return _cmd_info()
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "screen":
        return _cmd_screen(args)
    if args.command == "enumerate":
        return _cmd_enumerate(args)
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
