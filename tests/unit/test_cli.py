"""CLI contract: command set, version, verify on a tiny config, exit codes, documented options."""

import re
from pathlib import Path

import yaml
from typer.testing import CliRunner

from src.cli.main import app

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (ROOT / "specs" / "001-create-ddpm" / "contracts" / "cli.md").read_text(encoding="utf-8")
COMMANDS = ["verify", "train", "evaluate", "sample", "denoise-strip", "benchmark"]


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_help_lists_exactly_six_commands():
    out = _plain(runner.invoke(app, ["--help"], env={"COLUMNS": "200"}).output)
    listed = re.findall(r"│ (\S+)\s{2,}", out.split("Commands")[1])
    assert sorted(listed) == sorted(COMMANDS)


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Hands-on DDPM version: 0.1.0" in result.output


def test_verify_tiny_config(tiny_config, tmp_path):
    # With T=50 the default β range leaves ᾱ_T ≈ 0.60 (verify correctly fails); a short schedule needs
    # a larger β_end to reach noise. Mean/std check uses 256 images, so tolerance holds.
    tiny_config["diffusion"]["beta_end"] = 0.3  # ᾱ_T ≈ 2e-4 for T=50
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(tiny_config))
    result = runner.invoke(app, ["verify", "--config", str(path), "--skip-memory"])
    assert result.exit_code == 0, result.output
    assert "Checks passed: 6/6" in result.output


def test_unknown_key_exits_1(tiny_config, tmp_path):
    tiny_config["model"]["bogus"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(tiny_config))
    assert runner.invoke(app, ["verify", "--config", str(path), "--skip-memory"]).exit_code == 1


def test_missing_checkpoint_exits_1(tmp_path):
    assert runner.invoke(app, ["sample", "--checkpoint", str(tmp_path / "missing.pt")]).exit_code == 1


def test_help_options_match_contract():
    sections = re.split(r"^### 2\.\d+ `", CONTRACT, flags=re.M)[1:]
    for section in sections:
        name = section.split("`", 1)[0]
        if name not in COMMANDS:
            continue
        body = section.split("#### ")[0]
        documented = set(re.findall(r"--[a-z][a-z-]+", body.split("- **Options**")[1] if "- **Options**" in body else body))
        help_text = _plain(runner.invoke(app, [name, "--help"], env={"COLUMNS": "200"}).output)
        missing = sorted(opt for opt in documented if opt not in help_text)
        assert not missing, f"{name}: options in contracts/cli.md but not in --help: {missing}"
