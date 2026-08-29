"""Seed a project directory for the Playwright suite, from the real kernels.

Phase 1.1's end-to-end tests ran against the stub server, which served the
generated fixtures. #89 deleted it, so the suite needs a project that exists on
disk before the real server opens it. This builds one: the committed Tecator
data written out as a CSV and read back through the real reader and the real
import handler, the artboard's
four-branch pipeline written through the pipeline store, and every node
executed so its arrays are where `pipelines/current/state` looks for them.

Nothing here computes a number. The import goes through the same handler a
browser posts to, and the arrays come from `executor.execute`, so what the
screens read is what the application produces rather than a copy of it kept in
step by hand — which is the failure mode the fixtures had.

A state is a project, so a starting state is a seeded directory:

    uv run python tests/seed_e2e.py            <directory>   # imported and run
    uv run python tests/seed_e2e.py --empty    <directory>   # nothing in it
    uv run python tests/seed_e2e.py --unrun    <directory>   # imported, not run

`--fresh` removes the directory first, which is what the Playwright config
wants: a project left over from a previous run carries its arrays and its
edits. It is a flag rather than the default because this takes a path from the
command line and deletes it - a seed script should not be one typo away from
removing someone's project. Without it, `create_project` refuses a directory
that is not empty, which is the safe failure.

`--serve` seeds and then runs the server over what it seeded, and the directory
may be left out entirely - it comes from `CHEMOMETRICS_PROJECT`, which is the
same variable the server reads, so the two cannot disagree about which project
is open. That combination exists because of Windows: `playwright.config.ts` used
to chain `seed && server` in the `webServer` command, and Playwright hands that
string to `cmd.exe`, where it did not survive. Seed and server each ran
perfectly on their own there - proven by two smoke steps in CI - so the fix was
to stop needing a shell operator at all.

`--empty` is the state `EmptyProject` renders, reached by opening an empty
project rather than by a query parameter the server had to be taught to
understand. `--unrun` writes the pipeline and leaves every node without arrays,
so a run started there does real work and can be watched, cancelled and - on
the branch that cannot be fitted - failed. The default runs everything, which
is what the screens that only read want to find.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import sys
from pathlib import Path
from uuid import UUID


def tecator_csv() -> bytes:
    """The committed Tecator data, written as a file the CSV reader can read.

    `data/tecator/tecator.txt` is not a grid — it carries a prose header and
    the 22 principal components the file also supplies — so `load_tecator`
    parses it specially and no reader will take it. Rewriting it as a CSV means
    the seed goes in through the same reader a user's file does, on the same
    240 x 100 numbers, rather than around it.

    The wavelength axis is written as the header row, so it is *read* rather
    than reconstructed. `load_tecator` reconstructs it because the original
    file has no axis in it; a file that states its axis is the ordinary case
    and the one the import screen should be showing.
    """
    from chemometrics_workbench.datasets import load_tecator

    tecator = load_tecator()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["sample_id", *[f"{value:.4f}" for value in tecator.axis.values], *tecator.targets]
    )
    for index, sample_id in enumerate(tecator.sample_ids):
        writer.writerow(
            [
                sample_id,
                *[f"{value:.6f}" for value in tecator.spectra[index]],
                *[f"{tecator.targets[name][index]:.4f}" for name in tecator.targets],
            ]
        )
    return buffer.getvalue().encode("utf-8")


def synthetic_dataset(directory: Path, project, n_samples: int = 3000, n_variables: int = 1200):  # type: ignore[no-untyped-def]
    """A dataset big enough that a run takes long enough to watch.

    Tecator is 240 x 100, and every kernel in the pipeline is NumPy: the whole
    fourteen-node run finishes in about 0.2 s, which is real work and far too
    fast for a browser polling four times a second to catch a node in the
    `running` state. `#49` requires that state to be reachable, so the project
    the run tests use carries a bigger matrix.

    3,000 x 1,200 is about 1.5 s of work per branch here, so the whole run is
    ten-ish seconds - comfortably longer than the poll that has to see into it.
    It was 2,000 x 800, which was three seconds on the machine this was written
    on and under two on a macOS runner: fast enough that the run could be over
    before the assertion looking for a running node arrived. Sizing this for
    the slowest machine rather than the fastest is backwards.

    Generated, not measured, and **no number is claimed from it** - the same
    footing as #86's decimation fixture. It is shaped like spectra (a smooth
    baseline plus a few gaussian bands) so that SNV and Savitzky-Golay do the
    work they would do on a real file rather than degenerating on noise.
    """
    import numpy as np

    from chemometrics_workbench.models import (
        AxisKind,
        Dataset,
        DatasetVersion,
        SourceFile,
        VariableAxis,
    )
    from chemometrics_workbench.project import add_dataset, write_array

    axis = np.linspace(1000.0, 2500.0, n_variables)
    rng = np.random.default_rng(20260828)
    baseline = 0.6 + 0.25 * np.exp(-((axis - 1400.0) ** 2) / 90_000.0)
    bands = sum(
        rng.uniform(0.05, 0.35, (n_samples, 1)) * np.exp(-((axis - centre) ** 2) / (2 * width**2))
        for centre, width in ((1210.0, 28.0), (1720.0, 41.0), (2100.0, 35.0), (2310.0, 22.0))
    )
    spectra = baseline + bands + rng.normal(0.0, 0.002, (n_samples, n_variables))
    spectra *= rng.uniform(0.9, 1.1, (n_samples, 1))  # the scatter SNV is for

    array_path, content_hash = write_array(directory, spectra)
    dataset = Dataset(project_id=project.project_id, name="synthetic_wide", description="")
    version = DatasetVersion(
        dataset_id=dataset.dataset_id,
        version=1,
        content_hash=content_hash,
        n_samples=n_samples,
        n_variables=n_variables,
        axis=VariableAxis(kind=AxisKind.WAVELENGTH_NM, values=[float(v) for v in axis], unit="nm"),
        sample_ids=[f"S{index + 1:04d}" for index in range(n_samples)],
        targets={},
        source=SourceFile(
            filename="synthetic_wide.npy",
            file_hash=content_hash,
            reader="seed_e2e.synthetic_dataset",
            reader_version="1",
        ),
        array_path=array_path,
    )
    add_dataset(directory, dataset, version)
    return version


def build_pipeline(project_id: UUID, version_id: UUID):  # type: ignore[no-untyped-def]
    """Four preprocessing branches off one source, as the artboard draws it.

    The same graph the Phase 1.1 fixture generator built, kept because the screens
    were drawn against it and the walkthrough names its nodes. Branch D carries
    the split *downstream* of preprocessing and upstream of centring, which is
    the ordering `metrics-and-validation.md` §9 calls correct.
    """
    from chemometrics_workbench.models import (
        MSC,
        SNV,
        Autoscale,
        EstimatorNode,
        KFoldSplit,
        MeanCentre,
        PCASpec,
        Pipeline,
        PreprocessNode,
        SavitzkyGolay,
        SourceNode,
        SplitNode,
    )

    savgol = SavitzkyGolay(window_length=11, polyorder=2, deriv=1)
    return Pipeline(
        project_id=project_id,
        name="Scatter correction comparison",
        nodes=[
            SourceNode(id="source", version_id=version_id),
            # A: SNV
            PreprocessNode(id="snv", inputs=("source",), step=SNV()),
            PreprocessNode(id="centre_a", inputs=("snv",), step=MeanCentre()),
            EstimatorNode(id="pca_a", inputs=("centre_a",), spec=PCASpec(n_components=5)),
            # B: MSC
            PreprocessNode(id="msc", inputs=("source",), step=MSC()),
            PreprocessNode(id="centre_b", inputs=("msc",), step=MeanCentre()),
            EstimatorNode(id="pca_b", inputs=("centre_b",), spec=PCASpec(n_components=5)),
            # C: Savitzky-Golay first derivative
            PreprocessNode(id="savgol", inputs=("source",), step=savgol),
            PreprocessNode(id="autoscale_c", inputs=("savgol",), step=Autoscale()),
            EstimatorNode(id="pca_c", inputs=("autoscale_c",), spec=PCASpec(n_components=5)),
            # D: SNV then Savitzky-Golay - the path the exit criterion walks
            PreprocessNode(id="snv_savgol", inputs=("snv",), step=savgol),
            SplitNode(id="split_d", inputs=("snv_savgol",), spec=KFoldSplit(n_splits=10, seed=42)),
            PreprocessNode(id="centre_d", inputs=("split_d",), step=MeanCentre()),
            EstimatorNode(id="pca_d", inputs=("centre_d",), spec=PCASpec(n_components=5)),
        ],
    )


def failing_branch(start: float, end: float):  # type: ignore[no-untyped-def]
    """A branch that really fails, so the `failed` state is not a fixture.

    A handful of channels cannot yield twelve components, and
    `decomposition.py` refuses to return fewer by name — "N components were
    asked of a matrix of rank R" — which is the sentence the 1.1 stub imitated
    with `?failrun`. Now the kernel says it, and the canvas marks the node the
    executor named rather than the last one that reported progress.

    The window is taken from the dataset's own axis, so this works on whatever
    was seeded rather than assuming Tecator's 850-1050 nm.
    """
    from chemometrics_workbench.models import (
        EstimatorNode,
        MeanCentre,
        PCASpec,
        PreprocessNode,
        RangeSelect,
    )

    return [
        PreprocessNode(id="range_e", inputs=("source",), step=RangeSelect(start=start, end=end)),
        PreprocessNode(id="centre_e", inputs=("range_e",), step=MeanCentre()),
        EstimatorNode(id="pca_e", inputs=("centre_e",), spec=PCASpec(n_components=12)),
    ]


def seed(directory: Path, *, run: bool = True, failing: bool = False) -> None:
    """Import a dataset and write the pipeline; run it unless asked not to."""
    # Imported here rather than at module scope: `CHEMOMETRICS_PROJECT` has to
    # be set before `api` is imported, because the project path is read at
    # import time exactly as the token is.
    os.environ["CHEMOMETRICS_PROJECT"] = str(directory)
    from fastapi.testclient import TestClient

    from chemometrics_workbench import api, server
    from chemometrics_workbench.executor import execute, experiment_for
    from chemometrics_workbench.project import (
        create_project,
        open_project,
        write_experiment,
        write_pipeline,
    )

    # Created here rather than left to the server's first-use default, which
    # names a project after its directory. The screens show the project's name.
    if not (directory / "project.json").exists():
        create_project(directory, name="Tecator meat study", description="")
    project = open_project(directory)

    if failing:
        # The run tests need work that takes long enough to watch, which
        # Tecator does not; see `synthetic_dataset`.
        version = synthetic_dataset(directory, project)
    else:
        # Otherwise the import goes through the handler a browser posts to, so
        # the dataset is recorded exactly as a real one is - detection included.
        # Deliberately not `with TestClient(...)`: the context manager runs the
        # application's lifespan, and this app's shutdown calls
        # `JOBS.shutdown()`. Under `--serve` the seed and the server share one
        # process, so exiting that block would leave the server it is about to
        # start with a thread pool that has already been shut down - every run
        # then failing with "cannot schedule new futures after shutdown".
        # Nothing is needed from the lifespan here: this borrows the import
        # handler, it does not run a server.
        client = TestClient(server.app)
        response = client.post(
            "/api/import",
            files={"file": ("tecator.csv", tecator_csv())},
            data={"name": "tecator_raw"},
            headers={"Authorization": f"Bearer {server.TOKEN}"},
        )
        response.raise_for_status()
        entry = response.json()
        version = api._current_version(
            directory,
            build_pipeline(project.project_id, UUID(entry["versions"][-1]["version_id"])),
        )
        assert version is not None, "the import did not leave a dataset version behind"

    pipeline = build_pipeline(project.project_id, version.version_id)
    if failing:
        axis = version.axis.values
        # Six channels off the front of whatever axis was seeded. Six, not
        # twelve: a PCA that retains *all* the components it was asked for
        # fails too, but on the SPE limit having no residual to take a quantile
        # of - a different sentence, and not the one #49's failed state shows.
        pipeline = pipeline.model_copy(
            update={"nodes": [*pipeline.nodes, *failing_branch(axis[0], axis[5])]}
        )
    write_pipeline(directory, pipeline)

    computed = 0
    if run:
        finished = execute(directory, pipeline, version)
        write_experiment(directory, experiment_for(pipeline, version, finished))
        computed = len(finished.outputs)
    print(
        f"seeded {directory}: {version.n_samples} x {version.n_variables}, "
        f"{len(pipeline.nodes)} nodes, {computed} computed"
        + (", one branch that cannot be fitted" if failing else "")
    )


def main(argv: list[str]) -> int:
    modes = {"--empty", "--unrun"}
    flags = modes | {"--fresh", "--serve"}
    mode = next((argument for argument in argv if argument in modes), None)
    paths = [argument for argument in argv if argument not in flags]

    if len(paths) > 1:
        print(__doc__, file=sys.stderr)
        return 2
    if paths:
        directory = Path(paths[0]).resolve()
    elif os.environ.get("CHEMOMETRICS_PROJECT"):
        directory = Path(os.environ["CHEMOMETRICS_PROJECT"]).resolve()
    else:
        print(__doc__, file=sys.stderr)
        return 2

    if "--fresh" in argv and directory.exists():
        # `shutil` rather than `rm -rf`, because this runs on Windows too - the
        # reason the removal moved in here from the Playwright command at all.
        shutil.rmtree(directory)
    os.environ["CHEMOMETRICS_PROJECT"] = str(directory)
    if mode == "--empty":
        from chemometrics_workbench.project import create_project

        if not (directory / "project.json").exists():
            create_project(directory, name="Tecator meat study", description="")
        print(f"seeded {directory}: empty, as an unimported project is")
        return _serve(argv)

    seed(directory, run=mode != "--unrun", failing=mode == "--unrun")
    return _serve(argv)


def _serve(argv: list[str]) -> int:
    """Run the server over what was just seeded, if asked to."""
    if "--serve" not in argv:
        return 0
    from chemometrics_workbench import server

    server.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
