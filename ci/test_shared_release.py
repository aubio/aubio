"""Consumer authority remains explicit around the pinned shared transaction."""

import os
import re
import subprocess
import textwrap
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def test_release_workflow_preserves_identity_gates_and_same_run_artifacts() -> None:
    workflow = (ROOT / ".github/workflows/build.yml").read_text()
    job = workflow.split("\n  publish-release:\n", 1)[1]
    for required in (
        "needs: [plan, ci-passed]",
        "github.repository == 'LedFx/aubio-ledfx'",
        "github.event_name == 'push'",
        "startsWith(github.ref, 'refs/tags/v')",
        "needs.ci-passed.result == 'success'",
        "!cancelled()",
        "needs.plan.result == 'success'",
        "needs.plan.outputs.release == 'true'",
        "name: pypi",
        "queue: max",
        "cancel-in-progress: false",
        "pattern: wheels-*",
        "permission-contents: write",
        "permission-attestations: write",
    ):
        assert required in job
    assert "run-id:" not in job
    assert job.count("id-token: write") == 1
    assert "name: cibw-sdist" in job
    pins = re.findall(
        r"uses: LedFx/release-ci/actions/release@([0-9a-f]{40})(?:[ \t]+#.*)?[ \t]*$",
        job,
        re.MULTILINE,
    )
    assert len(pins) == 3 and len(set(pins)) == 1
    assert job.count("uses: LedFx/release-ci/actions/release@") == 3
    assert job.count("project: release-tools") == 3
    assert job.count("wheel-plan: ${{ needs.plan.outputs.wheel-plan }}") == 3
    planning = re.findall(
        r"uses: LedFx/release-ci/actions/plan@([0-9a-f]{40})(?:[ \t]+#.*)?[ \t]*$",
        workflow,
        re.MULTILINE,
    )
    assert planning == [pins[0]]
    assert "sparse-checkout-cone-mode: false" in job
    assert "pyproject.toml" in job
    assert "policy:" not in workflow
    assert (
        "uv run --frozen --only-group wheel-build python -m cibuildwheel ." in workflow
    )
    assert (
        '--config-file pyproject.toml --platform "$PLATFORM" --archs "$ARCH"'
        in workflow
    )
    assert "uses: pypa/cibuildwheel@" not in workflow
    assert (
        job.index("phase: prepare")
        < job.index("uses: actions/attest@")
        < job.index("phase: check-upload")
        < job.index("uses: pypa/gh-action-pypi-publish@")
        < job.index("phase: finalize")
    )
    assert "bundle-path" in job and "release-snapshot.json" in job
    assert "softprops" not in workflow and "--clobber" not in workflow


def test_pyproject_keeps_native_portable_matrix() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    rows = config["tool"]["release-ci"]["targets"]
    assert rows and len({(row["platform"], row["arch"]) for row in rows}) == len(rows)
    assert all({"runner", "platform", "arch"} <= set(row) for row in rows)
    dependencies = config["dependency-groups"]["wheel-build"]
    assert len(dependencies) == 1
    assert re.fullmatch(
        r"cibuildwheel(?:\[uv\])?==[0-9]+\.[0-9]+\.[0-9]+", dependencies[0]
    )
    assert config["project"]["name"] == "aubio-ledfx"
    assert set(config["tool"]["release-ci"]) == {"targets"}
    assert not (ROOT / ".github/release-policy.json").exists()


def test_upload_sidecars_leave_frozen_inputs_unchanged(tmp_path: Path) -> None:
    workflow = (ROOT / ".github/workflows/build.yml").read_text()
    match = re.search(
        r"(?m)^      - name: Stage verified distributions for PyPI\n"
        r"(?:(?:^        .*\n)|(?:^\n))*?^        run: \|\n"
        r"((?:^          .*\n|^\n)+)",
        workflow,
    )
    assert match is not None, "The uploader needs a separate verified input copy"
    stage = workflow.split("      - name: Stage verified distributions for PyPI\n", 1)[
        1
    ].split("\n      - ", 1)[0]
    assert "if: steps.upload.outputs.pypi_upload == 'true'" in stage
    assert "working-directory: ${{ github.workspace }}" in stage
    assert workflow.index("phase: check-upload") < workflow.index(
        "Stage verified distributions for PyPI"
    )
    uploader = workflow.split("uses: pypa/gh-action-pypi-publish@", 1)[1].split(
        "\n      - ", 1
    )[0]
    assert "packages-dir: pypi-dist/" in uploader
    script = textwrap.dedent(match.group(1))
    original = tmp_path / "dist"
    original.mkdir()
    frozen = {
        "example-1.0-py3-none-any.whl": b"tested wheel",
        "example-1.0.tar.gz": b"tested sdist",
    }
    for name, data in frozen.items():
        (original / name).write_bytes(data)
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_WORKSPACE": str(tmp_path)},
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    staging = tmp_path / "pypi-dist"
    assert {p.name: p.read_bytes() for p in staging.iterdir()} == frozen
    for name in frozen:
        (staging / (name + ".publish.attestation")).write_bytes(
            b"generated PyPI sidecar"
        )
    assert {p.name: p.read_bytes() for p in original.iterdir()} == frozen
    before_retry = {p.name: p.read_bytes() for p in staging.iterdir()}
    retry = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_WORKSPACE": str(tmp_path)},
        capture_output=True,
        check=False,
    )
    assert retry.returncode != 0
    assert {p.name: p.read_bytes() for p in staging.iterdir()} == before_retry
    assert {p.name: p.read_bytes() for p in original.iterdir()} == frozen


if __name__ == "__main__":
    test_release_workflow_preserves_identity_gates_and_same_run_artifacts()
    test_pyproject_keeps_native_portable_matrix()
    with TemporaryDirectory() as directory:
        test_upload_sidecars_leave_frozen_inputs_unchanged(Path(directory))
    print("3 shared publication contracts passed")
