"""What is wrong with a pipeline that will nonetheless run.

Everything `models.py` can refuse, it refuses: an odd Savitzky-Golay window, a
`start` above its `end`, a cycle in the graph. What it cannot refuse is a
recipe that is well formed, executes without complaint, and produces a number
that is quietly wrong. Those are here.

**Every warning in this module catches a mistake whose symptom is a plausible
result.** A leaked mean does not raise; it lowers RMSECV. A PLS model on
uncentred data does not raise; it spends its first component on the offset.
Neither is discoverable by looking at the output, which is why they are stated
before the run rather than diagnosed after it.

## They warn, they do not block

The pipeline is the record of what was done. Refusing to run it would make the
application the author of the analysis, and silently relocating a node — moving
a `MeanCentre` below the split "for" the user — would make the recipe a lie.
`metrics-and-validation.md` §9 is explicit: the validator warns and names the
node, and the warning travels into the experiment record so the number is never
read without it.

## Adding a third

Every warning here has a document behind it saying what the consequence is. A
rule that seems obviously right but is written down nowhere is a preference,
and preferences do not belong in something that annotates a scientist's work.
"""

from __future__ import annotations

from dataclasses import dataclass

from chemometrics_workbench.models import (
    MSC,
    Autoscale,
    MeanCentre,
    NodeId,
    Pipeline,
    PipelineNode,
    PLSDASpec,
    PLSRegressionSpec,
)

__all__ = [
    "LEAK_BEFORE_SPLIT",
    "PLS_WITHOUT_CENTRING",
    "PipelineWarning",
    "check_pipeline",
]

#: Codes rather than message matching, so a screen can style or filter a
#: warning without parsing the sentence a person reads.
#:
#: One code for the leak rather than one per step, because the consequence is
#: one thing: a parameter fitted on every sample, above a split, is fitted on
#: the held-out ones too. A screen filtering for "this pipeline leaks" wants
#: all of them. The *sentence* differs per step - saying "the mean" about a
#: reference spectrum would be wrong in a way a user would notice - and the
#: name says `fitted` rather than `centring` because MSC is not centring (#103).
LEAK_BEFORE_SPLIT = "fitted_upstream_of_split"
PLS_WITHOUT_CENTRING = "pls_without_centring"

#: What `pls-regression.md` §3 means by centring. `Autoscale` counts because it
#: subtracts the column means before it divides. **MSC is deliberately not
#: here**: it estimates a reference spectrum and regresses against it, which
#: leaves no intercept for PLS's first component to stop chasing.
_CENTRING = (MeanCentre, Autoscale)

#: The steps that estimate something *across samples*, so that one sample's
#: output depends on which others were present. That is the property §9's leak
#: rule is about, and it is wider than centring: `MSCTransformer` estimates its
#: reference - the mean or median spectrum - from the fit set.
#:
#: Everything else in the schema is row-wise or column-selecting and is
#: legitimate above a split: SNV, Normalise, SavitzkyGolay, BaselineCorrect,
#: RangeSelect. `test_a_step_that_estimates_nothing_may_sit_above_a_split`
#: asserts that silence.
_FITTED_ACROSS_SAMPLES = (MeanCentre, Autoscale, MSC)


@dataclass(frozen=True)
class PipelineWarning:
    """One thing worth saying about a pipeline before it is run.

    `node_id` is what the canvas points at. `related` names the other node that
    makes it a problem — the split that is being leaked into — because a
    warning about `centre_a` alone leaves the user hunting for which split it
    means in a graph with four branches.
    """

    code: str
    node_id: NodeId
    message: str
    related: tuple[NodeId, ...] = ()
    severity: str = "warning"


def check_pipeline(pipeline: Pipeline) -> list[PipelineWarning]:
    """Everything worth saying about `pipeline`, in the order its nodes appear.

    An empty list means there is nothing to tell the user, not that the recipe
    is good: this checks two named mistakes, and says so.
    """
    by_id = {node.id: node for node in pipeline.nodes}
    found: list[PipelineWarning] = []

    for node in pipeline.nodes:
        found.extend(_leak_before_split(node, by_id))
        found.extend(_pls_without_centring(node, by_id))
    return found


def _leak_before_split(
    node: PipelineNode, by_id: dict[NodeId, PipelineNode]
) -> list[PipelineWarning]:
    """`metrics-and-validation.md` §9: fitted before the split is fitted on everything.

    Savitzky-Golay, derivatives, SNV, range selection and unit conversion are
    legitimate above a split — they estimate nothing from the set of samples,
    so a sample's output does not depend on which other samples were present.
    Centring, autoscaling and MSC do, and the validation rows contribute to the
    statistics the training rows are then judged against.

    §9 names centring and autoscaling and leaves MSC off both of its lists.
    It belongs on this one: `MSCTransformer` estimates a reference spectrum -
    the mean or the median across the fit set - and regresses every sample
    against it, which is the same dependence on which samples were present.
    That is #103, and §9 now says so rather than leaving it to be re-derived.
    """
    if node.type != "preprocess" or not isinstance(node.step, _FITTED_ACROSS_SAMPLES):
        return []

    splits = sorted(other for other in _descendants(node.id, by_id) if by_id[other].type == "split")
    if not splits:
        return []

    named = ", ".join(repr(split) for split in splits)
    # The consequence in the step's own terms. A warning that says "the mean"
    # about a reference spectrum is wrong in a way a user would notice, and a
    # user who does not recognise the description will not act on the warning.
    if isinstance(node.step, MSC):
        leaked = (
            "The held-out spectra contribute to the reference spectrum the training "
            "spectra are corrected against, so a training sample's correction depends on "
            "samples it is about to be judged against, and RMSECV comes out optimistic."
        )
    else:
        leaked = (
            "The held-out samples contribute to the mean the training samples are centred "
            "by, and RMSECV comes out optimistic."
        )

    return [
        PipelineWarning(
            code=LEAK_BEFORE_SPLIT,
            node_id=node.id,
            related=tuple(splits),
            message=(
                f"{node.step.kind!r} at {node.id!r} is fitted above the split at {named}, so "
                f"it is fitted once on every sample. {leaked} "
                "Moving the node below the split fits it on each training fold instead."
            ),
        )
    ]


def _pls_without_centring(
    node: PipelineNode, by_id: dict[NodeId, PipelineNode]
) -> list[PipelineWarning]:
    """`pls-regression.md` §3: legal here, and almost always wrong.

    PLS fits the matrices it is given and centres nothing of its own, so
    centring is a node in the recipe or it has not happened.
    """
    if node.type != "estimator" or not isinstance(node.spec, PLSRegressionSpec | PLSDASpec):
        return []

    upstream = (by_id[other] for other in _ancestors(node.id, by_id))
    if any(other.type == "preprocess" and isinstance(other.step, _CENTRING) for other in upstream):
        return []

    return [
        PipelineWarning(
            code=PLS_WITHOUT_CENTRING,
            node_id=node.id,
            message=(
                f"{node.spec.kind!r} at {node.id!r} has no centring step above it. PLS fits "
                "the matrix it is given and centres nothing of its own, so the first "
                "component spends itself on the offset and the model carries no intercept to "
                "absorb it. Add a mean centre or an autoscale above this node."
            ),
        )
    ]


def _descendants(node_id: NodeId, by_id: dict[NodeId, PipelineNode]) -> set[NodeId]:
    """Every node that consumes this one's output, directly or through others."""
    children: dict[NodeId, list[NodeId]] = {nid: [] for nid in by_id}
    for node in by_id.values():
        for parent in node.inputs:
            children[parent].append(node.id)

    seen: set[NodeId] = set()
    stack = list(children[node_id])
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(children[current])
    return seen


def _ancestors(node_id: NodeId, by_id: dict[NodeId, PipelineNode]) -> set[NodeId]:
    """Every node this one's output is computed from."""
    seen: set[NodeId] = set()
    stack = list(by_id[node_id].inputs)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(by_id[current].inputs)
    return seen
