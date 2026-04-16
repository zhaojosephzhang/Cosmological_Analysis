#!/usr/bin/env python3
"""Minimal orchestration layer for the halo DM analysis suite.

This runner does not replace the scientific scripts. It standardizes:
- configuration
- working directory
- MPI wrapping
- step ordering
- dry-run / selective execution
"""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


# Pipeline steps are executed in this order when the user requests "all".
# Each step must have a matching block under the top-level "steps" key in config.yaml.
STEP_ORDER = ["prepare", "density_profile", "dm_impact", "map1d", "map2d"]


def shell_join(parts: List[str]) -> str:
    """Render a command list as a copy-pasteable shell command for logging."""
    return " ".join(shlex.quote(part) for part in parts)


def load_config(path: Path) -> Dict[str, Any]:
    """Load pipeline configuration from YAML or JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read config.yaml. "
            "Install it or provide a JSON config."
        ) from exc

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Top-level config must be a mapping: {path}")
    return data


def _as_list(value: Any) -> List[str]:
    """Normalize optional config values into a list of command-line strings."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def build_step_args(step_name: str, step_cfg: Dict[str, Any]) -> List[str]:
    """Translate the config block of one step into script-specific CLI arguments.

    Each scientific script has its own historical positional/optional arguments.
    This function is the adapter layer that keeps those script interfaces unchanged
    while allowing the pipeline runner to read a single config file.
    """
    args = step_cfg.get("args", {})
    if not isinstance(args, dict):
        raise ValueError(f"Step '{step_name}' args must be a mapping.")

    if step_name == "prepare":
        # data_halo_storing_with_stellar_MPI_input.py expects only snapshot_number.
        return [str(args["snapshot_number"])]

    if step_name == "density_profile":
        # This step is kept flexible: raw is passed through exactly as provided.
        return _as_list(args.get("raw", []))

    if step_name == "dm_impact":
        # DM_Impact_factor_morebin.py expects these positional values in order.
        # Lists are encoded as JSON strings so downstream scripts can parse them.
        ordered = [
            args.get("a"),
            args.get("h"),
            args.get("boxsize"),
            args.get("boxname"),
            args.get("factor_rv"),
        ]
        return [json.dumps(v) if isinstance(v, list) else str(v) for v in ordered if v is not None]

    if step_name == "map1d":
        # Halo_DM_1D_map_joblib_withstellar.py has two required positional args,
        # followed by several optional flags that are only emitted when configured.
        cli = [str(args["mass_range"]), str(args["halo_num"])]
        optional_flags = {
            "--snap-num": args.get("snap_num"),
            "--radial-bin-mode": args.get("radial_bin_mode"),
            "--agn-info": args.get("agn_info"),
            "--h5-write-mode": args.get("h5_write_mode"),
            "--mpi-io-mode": args.get("mpi_io_mode"),
        }
        for flag, value in optional_flags.items():
            if value is not None:
                cli.extend([flag, str(value)])
        return cli

    if step_name == "map2d":
        # The 2D map script uses positional optional arguments.
        cli = []
        if args.get("snap_num") is not None:
            cli.append(str(args["snap_num"]))
        if args.get("feedback_on") is not None:
            cli.append(str(args["feedback_on"]))
        return cli

    raise ValueError(f"Unknown step: {step_name}")


def build_command(
    repo_dir: Path,
    pipeline_cfg: Dict[str, Any],
    step_name: str,
    step_cfg: Dict[str, Any],
) -> List[str]:
    """Build the full command for one configured pipeline step.

    The returned command is a list suitable for subprocess.run().
    If use_mpi is true for the step, the command is wrapped as:
    mpirun -np <N> [extra_args...] python script.py ...
    """
    script = step_cfg.get("script")
    if not script:
        raise ValueError(f"Step '{step_name}' is missing 'script'.")

    python = str(pipeline_cfg.get("python", "python3"))
    workdir = pipeline_cfg.get("workdir", ".")
    script_path = repo_dir / workdir / script
    if not script_path.exists():
        raise FileNotFoundError(f"Script for step '{step_name}' not found: {script_path}")

    # Start with the plain Python command, then optionally wrap it in MPI below.
    cmd = [python, str(script_path)] + build_step_args(step_name, step_cfg)

    mpi_cfg = pipeline_cfg.get("mpi", {})
    if step_cfg.get("use_mpi", False):
        launcher = str(mpi_cfg.get("launcher", "mpirun"))
        np = str(mpi_cfg.get("np", 2))
        extra_args = _as_list(mpi_cfg.get("extra_args", []))
        cmd = [launcher, "-np", np] + extra_args + cmd

    return cmd


def selected_steps(requested: str) -> List[str]:
    """Return either the full ordered pipeline or a single requested step."""
    if requested == "all":
        return STEP_ORDER
    if requested not in STEP_ORDER:
        raise ValueError(f"Unsupported step: {requested}")
    return [requested]


def main() -> int:
    """Parse CLI arguments, load config, and execute enabled pipeline steps."""
    parser = argparse.ArgumentParser(description="Run the halo DM analysis pipeline.")
    parser.add_argument(
        "step",
        choices=["all"] + STEP_ORDER,
        help="Pipeline step to run, or 'all' to run enabled steps in order.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML/JSON pipeline config. Default: config.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    # repo_dir is the directory containing this runner, so relative script paths
    # in config.yaml are stable regardless of where the command is launched.
    repo_dir = Path(__file__).resolve().parent
    cfg = load_config(repo_dir / args.config if not Path(args.config).is_absolute() else Path(args.config))
    pipeline_cfg = cfg.get("pipeline", {})
    steps_cfg = cfg.get("steps", {})

    for step_name in selected_steps(args.step):
        step_cfg = steps_cfg.get(step_name)
        if not isinstance(step_cfg, dict):
            raise ValueError(f"Missing configuration for step '{step_name}'.")
        if not step_cfg.get("enabled", False):
            # Disabled steps are silently bypassed in "all" mode, but logged.
            print(f"[skip] {step_name}: disabled in config")
            continue

        cmd = build_command(repo_dir, pipeline_cfg, step_name, step_cfg)
        print(f"[run] {step_name}")
        print("      " + shell_join(cmd))
        sys.stdout.flush()

        if args.dry_run:
            continue

        # Use repo_dir as cwd so scripts with relative path assumptions behave
        # consistently whether launched interactively or from a batch job.
        proc = subprocess.run(cmd, cwd=repo_dir)
        if proc.returncode != 0:
            print(f"[fail] {step_name}: exit code {proc.returncode}", file=sys.stderr)
            return proc.returncode

    print("[done] pipeline runner finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
