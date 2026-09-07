import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .kineto_operator import KinetoOperator

# Prefixes used to identify the operator-level (as opposed to user/module-scoping) frames that are eligible to
# name a CPU launch segment. See `pick_segment_name` for details.
_ATEN_OR_C10D_PREFIXES = ("aten::", "c10d::")

# Synthetic thread ID used for the single merged CPU timeline. See `build_cpu_launch_segments`.
MERGED_TID = 0


@dataclass
class CpuLaunchSegment:
    """
    A single simplified CPU node spanning the time between two sequential kernel-launch operators (for example,
    `cudaLaunchKernel`).

    Rather than emitting one Chakra CPU node per Kineto CPU/user-annotation operator (which reflects the full,
    deeply nested call stack captured by the profiler), we only care about the CPU operator that actually fires
    a GPU operation, and how long the CPU spent between two such firings. This segment represents exactly that
    slice of time.

    All CPU ops (regardless of which OS thread recorded them) are merged into a single logical timeline before
    computing segments (see `build_cpu_launch_segments`), under the assumption that there is no genuine
    concurrency to model between them (e.g. a single forward/backward pass): every kernel-launch operator across
    every thread is treated as happening on one thread, and segment boundaries are the launch endpoints of that
    merged, globally-ordered sequence.

    Attributes
        tid (int): Always `MERGED_TID`, since all threads are merged into a single logical timeline.
        start_ts (int): Start timestamp of the segment. This is the end timestamp of the previous kernel-launch
            operator in the merged, globally-ordered sequence, or the earliest recorded timestamp for the first
            segment.
        end_ts (int): End timestamp of the segment. This is exactly the end timestamp (ts + dur) of `launch_op`,
            i.e. the moment the launch call that bounds this segment returns.
        name (str): Name assigned to this CPU node, chosen by `pick_segment_name`.
        launch_op (KinetoOperator): The kernel-launch operator (e.g. `cudaLaunchKernel`) whose end timestamp
            defines the end boundary of this segment.
        gpu_ops (List[KinetoOperator]): GPU-side operators (kernel/gpu_memcpy) launched by `launch_op`. These
            depend on this segment in the output Chakra trace.
    """

    tid: int
    start_ts: int
    end_ts: int
    name: str
    launch_op: KinetoOperator
    gpu_ops: List[KinetoOperator] = field(default_factory=list)

    @property
    def duration(self) -> int:
        """Wall-clock duration of this segment, i.e. the time between the two bounding launch endpoints."""
        return self.end_ts - self.start_ts


def _op_end(op: KinetoOperator) -> int:
    return op.timestamp + op.inclusive_dur


def _compute_enclosing_chains(
    cpu_ops: List[KinetoOperator], launch_ops: List[KinetoOperator]
) -> Tuple[Dict[int, List[KinetoOperator]], Dict[int, int]]:
    """
    Compute, for every launch op, the chain of CPU ops (outer to inner) that encloses it, and for every CPU op,
    how many distinct launch ops it encloses.

    This merges every CPU op across every original thread into a single call-stack-like timeline (under the
    assumption that there is no genuine concurrency to model, e.g. a single forward/backward pass, so the
    resulting merged sequence of intervals is properly nested). A single chronological sweep with a stack is
    used instead of pairwise interval-containment checks, so this runs in O(n log n) instead of O(n * m), which
    matters given there can be tens of thousands of CPU ops and thousands of launch ops.

    Args:
        cpu_ops (List[KinetoOperator]): All CPU (aten/user_annotation) operators, merged across every thread.
        launch_ops (List[KinetoOperator]): All kernel-launch operators, merged across every thread.

    Returns:
        Tuple[Dict[int, List[KinetoOperator]], Dict[int, int]]: A mapping from `id(launch_op)` to its enclosing
        chain (outer to inner), and a mapping from `id(cpu_op)` to the number of distinct launch ops it encloses.
    """
    # Event priorities control processing order for events sharing the same timestamp: an op ending exactly when
    # another starts must be popped first (so it doesn't spuriously enclose what comes next); a launch query at
    # the same timestamp as a CPU op's start must run after that op is pushed (so the op is treated as enclosing
    # a launch that starts at the exact same instant), and after any pop (so an op ending at that instant is not
    # treated as still open).
    POP, PUSH, QUERY = 0, 1, 2

    events: List[Tuple[int, int, int, str, KinetoOperator]] = []
    for i, op in enumerate(cpu_ops):
        events.append((op.timestamp, PUSH, i, "push", op))
        events.append((_op_end(op), POP, i, "pop", op))
    for i, op in enumerate(launch_ops):
        events.append((op.timestamp, QUERY, i, "query", op))

    events.sort(key=lambda e: (e[0], e[1]))

    stack: List[KinetoOperator] = []
    enclosing_chains: Dict[int, List[KinetoOperator]] = {}
    launch_count_by_op: Dict[int, int] = {}

    for _, _, _, kind, op in events:
        if kind == "pop":
            if stack and stack[-1] is op:
                stack.pop()
            elif op in stack:
                # Should not normally happen given well-nested input, but guard against malformed/overlapping
                # data rather than corrupting the whole stack.
                stack.remove(op)
        elif kind == "push":
            stack.append(op)
            launch_count_by_op.setdefault(id(op), 0)
        else:  # query
            chain = list(stack)
            enclosing_chains[id(op)] = chain
            for ancestor in chain:
                launch_count_by_op[id(ancestor)] = launch_count_by_op.get(id(ancestor), 0) + 1

    return enclosing_chains, launch_count_by_op


def pick_segment_name(
    launch_op: KinetoOperator,
    enclosing_chain: List[KinetoOperator],
    launch_count_by_op: Dict[int, int],
) -> str:
    """
    Choose the name of the CPU node ending at `launch_op`.

    Per the requested semantics, the name is the topmost (i.e. outermost) `aten::` or `c10d::` operator that
    encloses `launch_op` and that does not also enclose any *other* kernel-launch operator. Going any further up
    the ancestor chain would merge in time that "belongs" to a neighboring launch/segment, so we stop as soon as
    we find an ancestor that wraps this launch and only this launch.

    Args:
        launch_op (KinetoOperator): The kernel-launch operator that ends this segment.
        enclosing_chain (List[KinetoOperator]): CPU ops enclosing `launch_op`, ordered from outermost to
            innermost (as computed by `_compute_enclosing_chains`).
        launch_count_by_op (Dict[int, int]): Mapping from `id(cpu_op)` to the number of distinct launch ops it
            encloses (as computed by `_compute_enclosing_chains`).

    Returns:
        str: The chosen name. Falls back to the innermost aten/c10d ancestor if none uniquely wraps this single
        launch (e.g. a single `aten::` op that issues many kernel launches in a loop with no intervening
        aten/c10d frame), then to the innermost enclosing CPU op of any category, and finally to the launch
        operator's own name if nothing encloses it at all.
    """
    aten_or_c10d_enclosing = [op for op in enclosing_chain if op.name.startswith(_ATEN_OR_C10D_PREFIXES)]

    for candidate in aten_or_c10d_enclosing:  # already ordered outer -> inner
        if launch_count_by_op.get(id(candidate), 0) == 1:
            return candidate.name

    if aten_or_c10d_enclosing:
        # No aten/c10d ancestor uniquely wraps this launch. Best effort: use the innermost one.
        return aten_or_c10d_enclosing[-1].name

    if enclosing_chain:
        # No aten/c10d ancestor at all. Best effort: use the innermost enclosing CPU op of any category.
        return enclosing_chain[-1].name

    logging.debug(
        f"No enclosing CPU op found for launch op '{launch_op.name}' (ts={launch_op.timestamp}). Falling back "
        "to the launch operator's own name."
    )
    return launch_op.name


def build_cpu_launch_segments(
    kineto_tid_cpu_ops_map: Dict[int, List[KinetoOperator]],
    kineto_tid_launch_ops_map: Dict[int, List[KinetoOperator]],
    kineto_gpu_ops: List[KinetoOperator],
    kineto_thread_info: Dict[int, Tuple[int, int]],
) -> Dict[int, List[CpuLaunchSegment]]:
    """
    Build simplified CPU node segments, one per kernel-launch operator.

    All CPU ops and all kernel-launch operators are merged across every original thread into a single logical
    timeline (under the assumption that there is no genuine concurrency to model, e.g. a single forward/backward
    pass), rather than segmenting each thread independently. Each segment spans from the end of the previous
    kernel-launch operator in this merged, globally-ordered sequence (or the earliest recorded timestamp, for
    the very first segment) to the end of the kernel-launch operator that defines the segment. Every GPU
    operator launched by that kernel-launch operator becomes a dependent of the segment, which is exactly "the
    cudaLaunchKernel that launches the GPU node".

    Args:
        kineto_tid_cpu_ops_map (Dict[int, List[KinetoOperator]]): CPU (aten/user_annotation) operators grouped by
            thread ID.
        kineto_tid_launch_ops_map (Dict[int, List[KinetoOperator]]): Kernel-launch operators (e.g.
            `cudaLaunchKernel`) grouped by thread ID.
        kineto_gpu_ops (List[KinetoOperator]): All GPU-side operators (kernel/gpu_memcpy).
        kineto_thread_info (Dict[int, Tuple[int, int]]): Mapping from thread ID to (start_ts, end_ts) for that
            thread.

    Returns:
        Dict[int, List[CpuLaunchSegment]]: A single-entry mapping (keyed by `MERGED_TID`) to the list of
        segments in the merged timeline, sorted by end_ts.
    """
    all_launch_ops = [op for ops in kineto_tid_launch_ops_map.values() for op in ops]
    if not all_launch_ops:
        return {}

    all_cpu_ops = [op for ops in kineto_tid_cpu_ops_map.values() for op in ops]

    gpu_ops_by_correlation: Dict[int, List[KinetoOperator]] = {}
    for gpu_op in kineto_gpu_ops:
        gpu_ops_by_correlation.setdefault(gpu_op.correlation, []).append(gpu_op)

    sorted_launch_ops = sorted(all_launch_ops, key=_op_end)
    enclosing_chains, launch_count_by_op = _compute_enclosing_chains(all_cpu_ops, sorted_launch_ops)

    if kineto_thread_info:
        prev_end = min(start for start, _ in kineto_thread_info.values())
    else:
        prev_end = sorted_launch_ops[0].timestamp

    segments = []
    for launch_op in sorted_launch_ops:
        end_ts = _op_end(launch_op)
        # Guard against a pathological/overlapping launch op that would otherwise create a negative-duration
        # segment (e.g. duplicate or out-of-order launch events sharing near-identical timestamps).
        start_ts = min(prev_end, end_ts)

        name = pick_segment_name(launch_op, enclosing_chains.get(id(launch_op), []), launch_count_by_op)

        segment = CpuLaunchSegment(
            tid=MERGED_TID,
            start_ts=start_ts,
            end_ts=end_ts,
            name=name,
            launch_op=launch_op,
            gpu_ops=sorted(gpu_ops_by_correlation.get(launch_op.correlation, []), key=lambda op: op.timestamp),
        )
        segments.append(segment)
        prev_end = end_ts

    return {MERGED_TID: segments}
