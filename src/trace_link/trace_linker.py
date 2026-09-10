import bisect
import copy
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from et_replay.execution_trace import (
    EXECUTION_TRACE_PROCESS_ANNOTATION,
    EXECUTION_TRACE_THREAD_ANNOTATION,
)
from et_replay.execution_trace import Node as PyTorchOperator

from .chakra_device_trace_loader import ChakraDeviceTraceLoader
from .chakra_host_trace_loader import ChakraHostTraceLoader
from .cpu_launch_segmenter import CpuLaunchSegment, build_cpu_launch_segments
from .kineto_operator import KinetoOperator
from .unique_id_assigner import UniqueIdAssigner


class TraceLinker:
    """
    Links Chakra host execution traces (ET) and Chakra device ET to generate Chakra host + device ET.

    Attributes
        chakra_host_trace_loader (ChakraHostTraceLoader): Loader for Chakra host execution traces.
        chakra_device_trace_loader (ChakraDeviceTraceLoader): Loader for Chakra device execution traces.
        id_assigner (UniqueIdAssigner): Assigns unique IDs to operators.
    """

    def __init__(self) -> None:
        """Initialize the TraceLinker with a log level."""
        self.chakra_host_trace_loader = ChakraHostTraceLoader()
        self.chakra_device_trace_loader = ChakraDeviceTraceLoader()
        self.id_assigner = UniqueIdAssigner()

    def link(
        self,
        rank: int,
        chakra_host_trace: str,
        chakra_device_trace: str,
        output_file: str,
        strip_hierarchy: bool = False,
        sync_dependencies: bool = False,
    ) -> None:
        """
        Links Chakra host execution traces (ET) and Chakra device ET to generate Chakra host + device ET.

        Args:
            rank (int): Rank for the input traces.
            chakra_host_trace (str): Path to the Chakra host execution trace file.
            chakra_device_trace (str): Path to the Kineto trace file.
            output_file (str): Path for the output nyTorch execution trace plus file.
            strip_hierarchy (bool): If True, drop the original deeply nested host trace nodes from the output,
                keeping only the synthetic per-thread anchor, CpuLaunchSegment, and GPU nodes. See
                `construct_et_plus_data_from_segments` for details on why this is safe for comm_size/comm_type.
            sync_dependencies (bool): If True, encode cross-stream synchronization dependencies (e.g. a
                collective on one CUDA stream waiting on a compute kernel on another stream via
                cudaStreamWaitEvent) as extra `data_deps` edges between GPU nodes. Requires the device trace to
                have been captured with PyTorch's `enable_cuda_sync_events` experimental profiler flag; see
                `load_sync_dependencies_from_cuda_events`. If the device trace lacks "cuda_sync" events, this is
                a no-op (with a warning logged).
        """
        (
            kineto_cpu_ops,
            kineto_tid_ops_map,
            kineto_tid_cpu_ops_map,
            kineto_correlation_cuda_runtime_map,
            kineto_gpu_ops,
            kineto_id_arrow_op_map,
            kineto_id_cuda_launch_op_map,
            kineto_process_start_time,
            kineto_process_end_time,
            kineto_thread_debug,
            kineto_rf_id_to_device_op_map,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
            kineto_external_id_to_kineto_op_map,
            kineto_tid_launch_ops_map,
        ) = self.chakra_device_trace_loader.load(chakra_device_trace)

        if sync_dependencies:
            num_deps = self.load_sync_dependencies_from_cuda_events(chakra_device_trace, kineto_gpu_ops)
            if not num_deps:
                logging.warning(
                    f"--sync-dependencies was requested, but no 'cuda_sync' events were found in "
                    f"{chakra_device_trace}. The device trace must be captured with PyTorch's "
                    "enable_cuda_sync_events experimental profiler flag for this feature to have any effect."
                )

        kineto_tid_cpu_ops_map = self.enforce_inter_thread_order(kineto_tid_cpu_ops_map)

        # We only care about the CPU operator that actually fires a GPU operation (a kernel-launch operator such
        # as cudaLaunchKernel), and how long the CPU spent between two such firings. Segment the CPU timeline at
        # kernel-launch boundaries instead of emitting one Chakra CPU node per Kineto CPU op. All threads are
        # merged into a single logical timeline before segmenting (there is no genuine cross-thread concurrency
        # to model for a single forward/backward pass), so there is exactly one CPU node between each two
        # sequential kernel-launch operators, globally.
        segments_by_tid = build_cpu_launch_segments(
            kineto_tid_cpu_ops_map,
            kineto_tid_launch_ops_map,
            kineto_gpu_ops,
            kineto_thread_debug,
        )

        # Comm ops (e.g. "record_param_comms") carry the real tensor list (and therefore comm_size/comm_type)
        # for a collective or send/recv, but that data lives on the Chakra host trace node, not on the GPU
        # kernel's own Kineto event. Kineto correlates the two via a shared "External id": every op in the
        # nested CPU call stack that launched a GPU kernel -- including the comm op -- shares that GPU kernel's
        # external_id. Build external_id -> comm op so gpu_op_to_node can look up the matching host trace node.
        external_id_to_comm_kineto_op = self.build_external_id_to_comm_kineto_op_map(kineto_cpu_ops)

        chakra_execution_trace_plus_data = self.construct_et_plus_data_from_segments(
            chakra_host_trace, segments_by_tid, external_id_to_comm_kineto_op, strip_hierarchy
        )

        for node in chakra_execution_trace_plus_data["nodes"]:
            if "ncclDevKernel_SendRecv" in node.get("name", ""):
                if "dst_rank" not in node:
                    node["dst_rank"] = rank
                if "src_rank" not in node:
                    node["src_rank"] = rank
                if node["dst_rank"] == node["src_rank"]:
                    logging.warning(
                        f"SendRecv node '{node.get('name')}' (id={node.get('id')}) has "
                        f"dst_rank == src_rank == {node['dst_rank']}. This is likely incorrect."
                    )

        self.dump_chakra_execution_trace_plus(chakra_execution_trace_plus_data, output_file)

    def build_external_id_to_comm_kineto_op_map(
        self, kineto_cpu_ops: List[KinetoOperator]
    ) -> Dict[int, KinetoOperator]:
        """
        Map each communication-launching external_id to its "record_param_comms" Kineto CPU operator.

        "record_param_comms" is the operator PyTorch's profiler inserts around every collective/send/recv call,
        and it is the one that carries the full tensor list (and therefore the real comm_size) in the Chakra
        host execution trace. Other CPU ops sharing the same external_id (e.g. the dispatcher-level
        "c10d::allreduce_") are kept only as a fallback in case "record_param_comms" was not captured.

        Args:
            kineto_cpu_ops (List[KinetoOperator]): All Kineto CPU/user-annotation operators for this rank.

        Returns:
            Dict[int, KinetoOperator]: Mapping from external_id to the best-matching comm Kineto CPU operator.
        """
        external_id_to_comm_kineto_op: Dict[int, KinetoOperator] = {}
        for op in kineto_cpu_ops:
            if op.rf_id is None:
                continue
            if op.name == "record_param_comms" or op.external_id not in external_id_to_comm_kineto_op:
                external_id_to_comm_kineto_op[op.external_id] = op
        return external_id_to_comm_kineto_op

    def build_rf_id_to_host_node_map(self, nodes: List[Dict]) -> Dict[int, Dict]:
        """
        Map each Chakra host trace node's "rf_id" attribute to the node itself.

        "rf_id" (record function id) is the same identifier PyTorch's profiler stamps onto the corresponding
        Kineto CPU operator, so this map lets a Kineto operator be resolved back to its original, untouched host
        trace node (and, notably, that node's real tensor "inputs"/"outputs").

        Args:
            nodes (List[Dict]): The original host execution trace nodes.

        Returns:
            Dict[int, Dict]: Mapping from rf_id to host trace node.
        """
        rf_id_to_host_node = {}
        for node in nodes:
            for attr in node.get("attrs", []):
                if attr.get("name") == "rf_id":
                    rf_id_to_host_node[attr.get("value")] = node
                    break
        return rf_id_to_host_node

    def construct_et_plus_data_from_segments(
        self,
        chakra_host_trace: str,
        segments_by_tid: Dict[int, List[CpuLaunchSegment]],
        external_id_to_comm_kineto_op: Dict[int, KinetoOperator],
        strip_hierarchy: bool = False,
    ) -> Dict:
        """
        Construct the enhanced Chakra Host Execution Trace (ET+) data structure from CPU launch segments.

        Unlike the original ET+ construction, which enriches every node of the fully nested PyTorch host
        execution trace with Kineto timing data, this method keeps the original host trace nodes untouched (so
        nodes such as the process/thread annotations and communication-group metadata nodes, e.g.
        "## process_group:init ##", are preserved as-is, unless `strip_hierarchy` is set) but does **not** give
        the original deeply nested aten call-stack nodes any timing data. Instead, across the merged CPU
        timeline (all original threads combined, see `cpu_launch_segmenter.build_cpu_launch_segments`), it
        appends one new synthetic CPU node per `CpuLaunchSegment` -- i.e. one CPU node between each two
        sequential kernel-launch operators -- carrying the segment's name and duration, chained in execution
        order. Each GPU operator is attached as a dependent of the segment whose bounding kernel-launch operator
        fired it.

        Args:
            chakra_host_trace (str): Path to the Chakra host execution trace file.
            segments_by_tid (Dict[int, List[CpuLaunchSegment]]): CPU launch segments grouped by thread ID, as
                produced by `build_cpu_launch_segments`.
            external_id_to_comm_kineto_op (Dict[int, KinetoOperator]): Mapping from external_id to the
                "record_param_comms" (or best-effort fallback) Kineto CPU operator, as produced by
                `build_external_id_to_comm_kineto_op_map`.
            strip_hierarchy (bool): If True, drop the original deeply nested host trace nodes from the output
                entirely, keeping only the new synthetic per-thread anchor, `CpuLaunchSegment`, and GPU nodes.
                This is safe with respect to comm_size/comm_type: `gpu_op_to_node` already copies the real
                tensor list from the matching "record_param_comms" host node onto the GPU node itself before
                this method returns, so that data is preserved even once the original nodes are dropped. Note
                that this also drops "process_group:init" metadata nodes; only enable this if your downstream
                consumer does not depend on them.

        Returns:
            Dict: The constructed ET+ data.
        """
        logging.debug("Constructing ET+ data from CPU launch segments.")
        with open(chakra_host_trace, "r") as file:
            pytorch_et_data = json.load(file)

        existing_ids = [node["id"] for node in pytorch_et_data["nodes"]]
        # Seed the ID assigner above every existing host trace node ID so newly synthesized nodes never collide
        # with the untouched original nodes.
        self.id_assigner.next_id = max(existing_ids) + 1

        process_anchor_id = self.find_process_anchor_id(pytorch_et_data["nodes"])
        rf_id_to_host_node = self.build_rf_id_to_host_node_map(pytorch_et_data["nodes"])

        # Maps a GPU KinetoOperator to the new Chakra node ID it was emitted as, so that any GPU op referencing
        # it via `sync_dep` (see `load_sync_dependencies_from_cuda_events`) can resolve it to a concrete node
        # ID. Populated for every GPU op in a first pass below; `sync_dep` edges are only resolved into node IDs
        # in a second pass afterwards, since a cross-stream wait's producer kernel is not guaranteed to be
        # emitted before its consumer in this per-thread-launch-order timeline (e.g. under CUDA graph capture,
        # where the graph's kernels may be replayed out of the order they were originally captured/launched in).
        kineto_op_to_new_node_id: Dict[KinetoOperator, int] = {}
        # (gpu_op, gpu_node) pairs with a pending sync_dep, resolved into node["sync_dep"] in the second pass.
        pending_sync_deps: List[Tuple[KinetoOperator, Dict]] = []

        new_nodes = []
        for tid, segments in segments_by_tid.items():
            if not segments:
                continue

            # Kineto thread IDs (OS thread IDs) live in a different ID namespace than the host execution trace's
            # own "tid" attribute (a small per-process PyTorch thread index), so an existing host trace node
            # cannot be reused as the anchor for a given Kineto tid's segment chain. Instead, create a new
            # thread-annotation node per Kineto tid: its name matches the existing thread-annotation nodes, so
            # `PyTorchConverter.is_root_node` recognizes it as an independent traversal root, exactly like the
            # real per-thread root nodes already in the host trace.
            anchor_node = self.new_thread_anchor_node(tid, process_anchor_id)
            new_nodes.append(anchor_node)
            prev_node_id = anchor_node["id"]

            for segment in segments:
                segment_node = self.segment_to_node(segment, prev_node_id)
                new_nodes.append(segment_node)
                prev_node_id = segment_node["id"]

                for gpu_op in segment.gpu_ops:
                    gpu_node = self.gpu_op_to_node(
                        gpu_op, segment_node, external_id_to_comm_kineto_op, rf_id_to_host_node
                    )
                    new_nodes.append(gpu_node)

                    if gpu_op.sync_dep:
                        pending_sync_deps.append((gpu_op, gpu_node))

                    kineto_op_to_new_node_id[gpu_op] = gpu_node["id"]

        for gpu_op, gpu_node in pending_sync_deps:
            sync_dep_ids = []
            for producer_op in gpu_op.sync_dep:
                producer_node_id = kineto_op_to_new_node_id.get(producer_op)
                if producer_node_id is not None:
                    sync_dep_ids.append(producer_node_id)
                else:
                    logging.warning(
                        f"Sync dependency producer GPU op '{producer_op.name}' (external_id="
                        f"{producer_op.external_id}) was never emitted as a Chakra node; skipping this "
                        f"sync_dep edge for consumer '{gpu_op.name}' (external_id={gpu_op.external_id})."
                    )
            if sync_dep_ids:
                gpu_node["sync_dep"] = sync_dep_ids

        if strip_hierarchy:
            # Drop the original deeply nested host trace nodes entirely, keeping only the synthetic anchor,
            # segment, and GPU nodes. This is safe because gpu_op_to_node already copied any required
            # comm_size/comm_type inputs onto the GPU nodes above, before the original nodes are discarded here.
            pytorch_et_data["nodes"] = new_nodes
        else:
            pytorch_et_data["nodes"] += new_nodes

        logging.debug(
            f"Constructed ET+ data with {len(new_nodes)} new nodes from "
            f"{sum(len(segments) for segments in segments_by_tid.values())} CPU launch segments."
        )
        return pytorch_et_data

    def find_process_anchor_id(self, nodes: List[Dict]) -> int:
        """
        Find the host trace node ID of the (single) process-annotation node.

        New synthetic per-Kineto-tid thread-annotation nodes are parented under this node, mirroring how the
        real thread-annotation nodes are parented in the original host execution trace.

        Args:
            nodes (List[Dict]): The original host execution trace nodes.

        Returns:
            int: The host trace node ID of the process-annotation node. Falls back to the smallest existing node
                ID if no process-annotation node is found (should not normally happen).
        """
        for node in nodes:
            if node.get("name") == EXECUTION_TRACE_PROCESS_ANNOTATION:
                return node["id"]
        logging.warning(
            f"No '{EXECUTION_TRACE_PROCESS_ANNOTATION}' node found in the host execution trace. Falling back to "
            "the smallest existing node ID as the anchor for new thread-annotation nodes."
        )
        return min(node["id"] for node in nodes)

    def new_thread_anchor_node(self, tid: int, parent_id: int) -> Dict:
        """
        Create a new thread-annotation node anchoring a chain of CPU launch segments.

        Its name matches the real thread-annotation nodes already present in the host execution trace
        (`EXECUTION_TRACE_THREAD_ANNOTATION`), so `PyTorchConverter.is_root_node` recognizes it as an independent
        traversal root and processes these CPU launch segments as their own ordered chain, exactly as the real
        per-thread root nodes are already handled for the original, untouched host-trace nodes. Since all
        threads are merged into a single logical timeline before segmenting, there is only ever one such anchor
        node (`tid` is `cpu_launch_segmenter.MERGED_TID`), rather than one per original Kineto thread ID.

        Args:
            tid (int): Identifier for this anchor's segment chain (`cpu_launch_segmenter.MERGED_TID`).
            parent_id (int): The node ID of the host trace's process-annotation node.

        Returns:
            Dict: A node dict compatible with the Chakra host execution trace JSON schema.
        """
        node_id = self.id_assigner.generate_new_id()
        return {
            "id": node_id,
            "name": EXECUTION_TRACE_THREAD_ANNOTATION,
            "ctrl_deps": parent_id,
            "inputs": {"values": [], "shapes": [], "types": [], "strides": []},
            "outputs": {"values": [], "shapes": [], "types": [], "strides": []},
            "attrs": self._new_node_attrs(tid),
        }

    @staticmethod
    def _new_node_attrs(tid: int) -> List[Dict]:
        """Build the minimal "attrs" list required by the Chakra converter for a synthetic node."""
        return [
            {"name": "rf_id", "type": "uint64", "value": 0},
            {"name": "fw_parent", "type": "uint64", "value": 0},
            {"name": "seq_id", "type": "int64", "value": -1},
            {"name": "scope", "type": "uint64", "value": 0},
            {"name": "tid", "type": "uint64", "value": tid},
            {"name": "fw_tid", "type": "uint64", "value": 0},
            {"name": "op_schema", "type": "string", "value": ""},
        ]

    def segment_to_node(self, segment: CpuLaunchSegment, ctrl_dep: int) -> Dict:
        """
        Convert a `CpuLaunchSegment` into a Chakra host trace CPU node dict.

        Args:
            segment (CpuLaunchSegment): The CPU launch segment to convert.
            ctrl_dep (int): The node ID of the preceding node in execution order on this thread (either the
                previous segment, or the thread's anchor node for the first segment).

        Returns:
            Dict: A node dict compatible with the Chakra host execution trace JSON schema.
        """
        node_id = self.id_assigner.generate_new_id()
        return {
            "id": node_id,
            "name": segment.name,
            "ctrl_deps": ctrl_dep,
            "inputs": {"values": [], "shapes": [], "types": [], "strides": []},
            "outputs": {"values": [], "shapes": [], "types": [], "strides": []},
            "attrs": self._new_node_attrs(segment.tid),
            "ts": segment.start_ts,
            "inclusive_dur": segment.duration,
            "exclusive_dur": segment.duration,
        }

    def gpu_op_to_node(
        self,
        gpu_op: KinetoOperator,
        segment_node: Dict,
        external_id_to_comm_kineto_op: Dict[int, KinetoOperator],
        rf_id_to_host_node: Dict[int, Dict],
    ) -> Dict:
        """
        Convert a Kineto GPU operator into a Chakra host trace GPU node dict, dependent on `segment_node`.

        Args:
            gpu_op (KinetoOperator): The GPU-side Kineto operator (kernel/gpu_memcpy).
            segment_node (Dict): The CPU launch segment node this GPU operator depends on, i.e. the node
                representing the kernel-launch operator that fired it.
            external_id_to_comm_kineto_op (Dict[int, KinetoOperator]): Mapping from external_id to the comm
                Kineto CPU operator that launched it, as produced by `build_external_id_to_comm_kineto_op_map`.
            rf_id_to_host_node (Dict[int, Dict]): Mapping from rf_id to host trace node, as produced by
                `build_rf_id_to_host_node_map`.

        Returns:
            Dict: A node dict compatible with the Chakra host execution trace JSON schema.
        """
        node_id = self.id_assigner.generate_new_id()
        gpu_node = copy.deepcopy(segment_node)
        gpu_node.update(
            {
                "id": node_id,
                "ctrl_deps": segment_node["id"],
                "name": gpu_op.name,
                "cat": gpu_op.category,
                "ph": gpu_op.phase,
                "ts": gpu_op.timestamp,
                "inclusive_dur": gpu_op.inclusive_dur,
                "exclusive_dur": gpu_op.exclusive_dur,
                "stream": gpu_op.stream,
                **({"pg_name": gpu_op.pg_name} if gpu_op.is_inter_gpu_comms_op() and gpu_op.pg_name is not None else {}),
                **(
                    {"dst_rank": gpu_op.dst_rank}
                    if "ncclDevKernel_SendRecv" in gpu_op.name and getattr(gpu_op, "dst_rank", None) is not None
                    else {}
                ),
                **(
                    {"src_rank": gpu_op.src_rank}
                    if "ncclDevKernel_SendRecv" in gpu_op.name and getattr(gpu_op, "src_rank", None) is not None
                    else {}
                ),
            }
        )

        if gpu_op.is_inter_gpu_comms_op():
            comm_kineto_op = external_id_to_comm_kineto_op.get(gpu_op.external_id)
            comm_host_node = (
                rf_id_to_host_node.get(comm_kineto_op.rf_id)
                if comm_kineto_op is not None and comm_kineto_op.rf_id is not None
                else None
            )
            if comm_host_node is not None:
                # Restore the real tensor list from the comm op's host trace node, so the Chakra converter can
                # derive comm_size/comm_type for this COMM_COLL_NODE/COMM_SEND_NODE/COMM_RECV_NODE instead of
                # falling back to this synthetic node's always-empty inputs.
                gpu_node["inputs"] = comm_host_node["inputs"]
                gpu_node["outputs"] = comm_host_node["outputs"]
            else:
                logging.warning(
                    f"No comm host trace node found for GPU op '{gpu_op.name}' (external_id="
                    f"{gpu_op.external_id}). comm_size/comm_type will be missing for this node."
                )

        return gpu_node

    def load_sync_dependencies_from_cuda_events(
        self, chakra_device_trace: str, kineto_gpu_ops: List[KinetoOperator]
    ) -> int:
        """
        Populate cross-stream synchronization dependencies directly from Kineto's "cuda_sync"/"cuda_event"
        activity categories.

        These categories are only emitted when the trace was captured with PyTorch's
        `enable_cuda_sync_events` experimental profiler flag (i.e.
        `torch.profiler.profile(experimental_config=torch._C._profiler._ExperimentalConfig(
        enable_cuda_sync_events=True))`). When enabled, each "Stream Wait Event" ("cuda_sync") entry directly
        records the consuming stream (`stream`), `wait_on_stream` (the producer stream), and
        `wait_on_cuda_event_record_corr_id` (the correlation ID of the CUDA runtime call, e.g. cudaEventRecord,
        that recorded the CUDA event being waited on); each "cuda_event" entry records the stream a given CUDA
        event was recorded on. Together these let us deterministically identify, for every stream-to-stream
        wait, both endpoints of the dependency as actual GPU kernels already present in `kineto_gpu_ops`:

        * the producer: the last GPU op issued on the producer stream at or before the moment its completion was
          captured by the event record.
        * the consumer: the GPU op launched by the first kernel-launch CUDA runtime/driver call (e.g.
          cudaLaunchKernel, cuLaunchKernelEx, cudaMemcpyAsync, ...) issued on the same CPU thread, strictly after
          the `cudaStreamWaitEvent` call that produced this "Stream Wait Event" entry, whose launched GPU op lands
          on this wait's own consumer stream. This is a CPU-program-order fact, not a GPU-timestamp guess:
          PyTorch always issues `cudaStreamWaitEvent(consumer_stream, event)` from the same host thread that will
          go on to launch the kernel it is meant to gate. We don't just take the very next launch call
          unconditionally, though: when several stream waits are issued back-to-back on one host thread (e.g. one
          thread gating streams A, B, C in a row), unrelated launches for other streams can legitimately appear
          first, so we scan forward for the first launch matching this wait's own consumer stream instead. This
          replaces an earlier,
          less reliable approach that bisected the consumer stream's GPU-op timestamps against the "Stream Wait
          Event" entry's own `ts` (or `max(ts, record_ts)`): that entry's `ts` marks when the wait was merely
          *enqueued*, not when it actually released, and other GPU ops on the same stream are frequently
          interleaved in real time between those two moments (observed in ~80% of wait events on one workload),
          which could make the bisection land on an op that was not actually gated by this wait at all.

        This is far more direct than routing through Holistic Trace Analysis's critical path analysis (see
        `load_sync_dependencies`), which requires this same "cuda_sync" data anyway and is a much heavier,
        version-fragile dependency (its `critical_path_analysis` was found to crash on newer pandas versions in
        practice). For each identified dependency, the producer `KinetoOperator` is appended to the consumer
        `KinetoOperator`'s `sync_dep` list, so that `gpu_op_to_node` can later resolve it to a `sync_dep` edge
        (a `data_deps` entry, once converted) on the consumer's Chakra node, pointing at the producer's node ID.

        Args:
            chakra_device_trace (str): Path to the Kineto trace file.
            kineto_gpu_ops (List[KinetoOperator]): GPU-side Kineto operators already parsed from the same trace.
                Mutated in place: matched consumer ops get their `sync_dep` list populated with producer ops.

        Returns:
            int: The number of cross-stream synchronization dependencies found and recorded.
        """
        with open(chakra_device_trace, "r") as f:
            trace_events = json.load(f)["traceEvents"]

        # correlation ID of the CUDA runtime call that recorded a given CUDA event -> (stream, timestamp)
        event_record_by_corr: Dict[int, Tuple[Optional[int], float]] = {}
        stream_wait_events: List[Dict] = []
        # correlation ID of the "cudaStreamWaitEvent" CPU call -> that raw CPU event dict.
        cpu_wait_event_by_corr: Dict[int, Dict] = {}
        # (tid) -> list of raw CPU events that either launch a GPU op or are a "cudaStreamWaitEvent" call,
        # sorted by ts, so that "the first kernel-launch call strictly after a given cudaStreamWaitEvent call
        # on the same thread" can be found by a simple forward scan from that call's position in the list.
        launch_and_wait_events_by_tid: Dict[int, List[Dict]] = {}
        launch_op_names = {
            "cuLaunchKernel",
            "cuLaunchKernelEx",
            "cudaLaunchKernel",
            "cudaLaunchKernelExC",
            "cudaLaunchCooperativeKernel",
            "cudaMemcpy",
            "cudaMemcpyAsync",
            "cudaMemcpyFromSymbol",
            "cudaMemcpyToSymbol",
            "cudaMemsetAsync",
            "hipLaunchKernel",
            "hipExtLaunchKernel",
            "hipExtModuleLaunchKernel",
            "hipModuleLaunchKernel",
            "hipMemcpyWithStream",
            "hipMemcpyAsync",
        }
        for event in trace_events:
            args = event.get("args", {})
            cat = event.get("cat")
            if cat == "cuda_event":
                correlation = args.get("correlation")
                if correlation is not None:
                    event_record_by_corr[correlation] = (args.get("stream"), event.get("ts"))
            elif cat == "cuda_sync" and args.get("cuda_sync_kind") == "Stream Wait Event":
                stream_wait_events.append(event)
            elif cat in ("cuda_runtime", "cuda_driver") and event.get("tid") is not None:
                if event.get("name") == "cudaStreamWaitEvent":
                    correlation = args.get("correlation")
                    if correlation is not None:
                        cpu_wait_event_by_corr[correlation] = event
                    launch_and_wait_events_by_tid.setdefault(event["tid"], []).append(event)
                elif event.get("name") in launch_op_names:
                    launch_and_wait_events_by_tid.setdefault(event["tid"], []).append(event)
        for tid_events in launch_and_wait_events_by_tid.values():
            tid_events.sort(key=lambda e: e.get("ts", 0))

        # correlation ID of the CUDA runtime/driver launch call -> the GPU op it launched.
        gpu_op_by_corr: Dict[int, KinetoOperator] = {}
        for op in kineto_gpu_ops:
            if op.correlation is not None and op.correlation >= 0:
                gpu_op_by_corr[op.correlation] = op

        # stream -> GPU ops issued on that stream, sorted by timestamp, used to find "the last GPU op issued
        # on this stream at or before a given timestamp" (producer lookup) via binary search.
        gpu_ops_by_stream: Dict[int, List[KinetoOperator]] = {}
        for op in kineto_gpu_ops:
            if op.stream is not None:
                gpu_ops_by_stream.setdefault(op.stream, []).append(op)
        for ops in gpu_ops_by_stream.values():
            ops.sort(key=lambda op: op.timestamp)
        gpu_op_ts_by_stream = {stream: [op.timestamp for op in ops] for stream, ops in gpu_ops_by_stream.items()}

        num_deps = 0
        for wait_event in stream_wait_events:
            args = wait_event.get("args", {})
            wait_external_id = args.get("External id")
            consumer_stream = args.get("stream")
            wait_corr_id = args.get("correlation")
            record_corr_id = args.get("wait_on_cuda_event_record_corr_id")
            if wait_external_id is None or record_corr_id is None or consumer_stream is None or wait_corr_id is None:
                continue

            record = event_record_by_corr.get(record_corr_id)
            if record is None:
                logging.warning(
                    f"Stream Wait Event (external_id={wait_external_id}) references cuda event record "
                    f"correlation {record_corr_id}, but no matching 'cuda_event' entry was found."
                )
                continue
            producer_stream, record_ts = record

            producer_ops = gpu_ops_by_stream.get(producer_stream)
            producer_op_ts = gpu_op_ts_by_stream.get(producer_stream)
            if not producer_ops:
                continue
            producer_index = bisect.bisect_right(producer_op_ts, record_ts) - 1
            if producer_index < 0:
                continue
            producer_op = producer_ops[producer_index]

            # Find the CPU-side cudaStreamWaitEvent call for this wait (same correlation ID), then walk
            # forward in that thread's CPU-program-order event list for the first kernel-launch call whose
            # GPU op lands on this wait's own consumer stream: that GPU op is the consumer, i.e. the kernel
            # actually gated by this wait. We cannot simply take the very next launch call unconditionally --
            # when several stream waits are issued back-to-back on the same host thread (e.g. one thread
            # gating streams A, B, C in a row), the launches that satisfy each wait are not necessarily
            # interleaved 1:1 with the waits: unrelated kernel launches for other streams (already-waited-on
            # or not) can appear first. Scanning for the matching stream (rather than stopping at the first
            # launch of any kind) reliably finds the correct GPU op in that case.
            cpu_wait_op = cpu_wait_event_by_corr.get(wait_corr_id)
            if cpu_wait_op is None:
                continue
            tid_events = launch_and_wait_events_by_tid.get(cpu_wait_op["tid"])
            if not tid_events:
                continue
            wait_index = next((i for i, e in enumerate(tid_events) if e is cpu_wait_op), None)
            if wait_index is None:
                continue
            consumer_op = None
            for candidate in tid_events[wait_index + 1 :]:
                if candidate.get("name") == "cudaStreamWaitEvent":
                    continue
                launch_corr = candidate.get("args", {}).get("correlation")
                candidate_op = gpu_op_by_corr.get(launch_corr) if launch_corr is not None else None
                if candidate_op is not None and candidate_op.stream == consumer_stream:
                    consumer_op = candidate_op
                    break
            if consumer_op is None:
                logging.warning(
                    f"Stream Wait Event (external_id={wait_external_id}) expected consumer stream "
                    f"{consumer_stream}, but no later kernel-launch call on the same host thread launched a GPU "
                    "op on that stream; skipping."
                )
                continue

            if consumer_op is producer_op or producer_op in consumer_op.sync_dep:
                continue

            # Sanity-check causality: a genuine dependency requires the producer to have finished (its own GPU
            # execution window ends) at or before the consumer starts. Under CUDA graph capture/replay, the
            # "cuda_sync"/"cuda_event" timestamps used above to identify *which* kernel is on either end of a
            # wait do not always line up with that kernel's actual per-replay execution window (the same op may
            # be replayed multiple times, and event/wait bookkeeping timestamps can reflect a different replay
            # instance than the GPU kernel timestamps do). When that happens, bisecting on those timestamps can
            # land on the wrong instance of a repeated kernel, producing a "dependency" that is backwards in time
            # (producer finishes after consumer already started) -- which is physically impossible for a real
            # dependency and must not be encoded. Drop those as unresolvable/spurious rather than emit a
            # contradictory edge.
            producer_end = producer_op.timestamp + producer_op.inclusive_dur
            if producer_end > consumer_op.timestamp:
                logging.warning(
                    f"Dropping implausible sync dep (from cuda_sync events): producer GPU op '{producer_op.name}' "
                    f"(stream {producer_stream}, external_id {producer_op.external_id}, ends at {producer_end}) "
                    f"would finish after consumer GPU op '{consumer_op.name}' (stream {consumer_stream}, "
                    f"external_id {consumer_op.external_id}, starts at {consumer_op.timestamp}) already started; "
                    "likely a CUDA graph replay timestamp mismatch."
                )
                continue

            consumer_op.sync_dep.append(producer_op)
            num_deps += 1
            logging.info(
                f"Sync dep (from cuda_sync events): producer GPU op '{producer_op.name}' (stream "
                f"{producer_stream}, external_id {producer_op.external_id}) -> consumer GPU op "
                f"'{consumer_op.name}' (stream {consumer_stream}, external_id {consumer_op.external_id})"
            )

        return num_deps

    def load_sync_dependencies(
        self, rank: int, kineto_file: str, annotation: str = "ProfilerStep", instance_id: int = 0
    ) -> Dict[int, List[int]]:
        """
        Load synchronization dependencies using Holistic Trace Analysis (HTA).

        Holistic Trace Analysis (HTA) provides various features for trace analysis, one of which is critical path
        analysis. This feature identifies dependencies between GPU and CPU operators that are in the critical path.
        This method leverages HTA's critical path analysis to determine synchronization points and dependencies,
        returning them as a dictionary.

        Args:
            rank (int): Rank for the input Kineto trace.
            kineto_file (str): Path to the Kineto trace file.
            annotation (str): Annotation to use for the analysis. Defaults to "ProfilerStep".
            instance_id (int): Instance ID for the analysis. Defaults to 0.

        Returns:
            Dict[int, List[int]]: A dictionary mapping end event's external ID to a list of start event's external IDs
                that have synchronization dependencies.
        """
        # Imported lazily so that importing this module (and the rest of the trace-linking pipeline) does not
        # require Holistic Trace Analysis (HTA) to be installed unless this optional feature is actually used.
        from hta.analyzers.critical_path_analysis import CPEdgeType
        from hta.trace_analysis import TraceAnalysis

        sync_dependencies = {}
        absolute_kineto_file = os.path.abspath(kineto_file)
        trace_dir = os.path.dirname(absolute_kineto_file)
        trace_analysis = TraceAnalysis(trace_dir=trace_dir, trace_files={rank: kineto_file})
        try:
            cp_graph, success = trace_analysis.critical_path_analysis(
                rank=rank, annotation=annotation, instance_id=instance_id
            )
            if not success:
                logging.error("Critical path analysis completed but failed to load Critical Path Graph.")
                return sync_dependencies

        except ValueError as e:
            logging.error("Critical path analysis encountered an invalid graph structure: %s", e)
            # Optionally, you could log more details or include rank-specific information if relevant
            return sync_dependencies

        raw_events = trace_analysis.t.get_raw_trace_for_one_rank(rank=rank)["traceEvents"]
        for edge in cp_graph.critical_path_edges_set:
            if edge.type in [CPEdgeType.SYNC_DEPENDENCY]:
                start_event_id, end_event_id = cp_graph.get_events_for_edge(edge)
                start_event, end_event = raw_events[start_event_id], raw_events[end_event_id]
                if "External id" in end_event["args"] and "External id" in start_event["args"]:
                    start_event_external_id = start_event["args"]["External id"]
                    end_event_external_id = end_event["args"]["External id"]
                    start_event_name = start_event["name"]
                    end_event_name = end_event["name"]
                    if start_event_external_id != end_event_external_id:
                        logging.info(
                            f"Sync dep: start_event_id {start_event_id}, end_event_id {end_event_id}, "
                            f"start_ext_id {start_event_external_id}, end_ext_id {end_event_external_id}, "
                            f"start_event_name '{start_event_name}', end_event_name '{end_event_name}'"
                        )
                        sync_dependencies.setdefault(end_event_external_id, []).append(start_event_external_id)
                else:
                    logging.warning(
                        f"Synchronization dependency from event {start_event_id} to event {end_event_id} will "
                        "not be considered due to missing external IDs."
                    )

        return sync_dependencies

    def enforce_inter_thread_order(
        self, kineto_tid_cpu_ops_map: Dict[int, List[KinetoOperator]], threshold: int = 1000
    ) -> Dict[int, List[KinetoOperator]]:
        """
        Enforce order between groups of operators in different threads.

        In Kineto traces with multiple threads, operators are executed in turns, creating groups. This function
        identifies these groups by detecting significant gaps in execution within each thread. It then establishes
        dependencies between these groups across different threads, ensuring the final Chakra execution traces reflect
        inter-thread dependencies realistically.

        An isolated group is formed when there's a significant gap in execution within a thread. Each new group relies
        on the last CPU operator from other threads, enforcing order and dependency across threads.

        Args:
            kineto_tid_cpu_ops_map (Dict[int, List[KinetoOperator]]): Kineto CPU operators grouped by thread ID.
            threshold (int): Threshold for significant gap detection in microseconds, used to define group boundaries.

        Returns:
            Dict[int, List[KinetoOperator]]: Updated map with enforced inter-thread order.
        """
        logging.debug("Enforcing inter-thread order in Kineto traces.")

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.process_thread_inter_thread_order, tid, ops, kineto_tid_cpu_ops_map, threshold
                ): tid
                for tid, ops in kineto_tid_cpu_ops_map.items()
            }

            for future in as_completed(futures):
                tid = futures[future]
                future.result()
                logging.debug(f"Thread {tid} dependencies processed.")

        return kineto_tid_cpu_ops_map

    def process_thread_inter_thread_order(
        self, tid: int, ops: List[KinetoOperator], ops_by_tid: Dict[int, List[KinetoOperator]], threshold: int
    ) -> None:
        """
        Process a single thread's operators to enforce inter-thread order.

        Args:
            tid (int): Thread ID.
            ops (List[KinetoOperator]): List of Kineto operators for the thread.
            ops_by_tid (Dict[int, List[KinetoOperator]]): Kineto operators grouped by thread ID.
            threshold (int): Threshold for significant gap detection in microseconds.
        """
        logging.debug(f"Thread {tid}: Identifying gaps for dependency linking with threshold {threshold}us.")
        sorted_ops = sorted(ops, key=lambda op: op.timestamp)
        last_cpu_node_rf_id = None

        for i, op in enumerate(sorted_ops):
            if (
                i == 0
                or (sorted_ops[i].timestamp - sorted_ops[i - 1].timestamp - sorted_ops[i - 1].inclusive_dur) > threshold
            ):
                last_cpu_node_rf_id = self.find_last_cpu_node_before_timestamp(ops_by_tid, tid, op.timestamp)
                if last_cpu_node_rf_id:
                    logging.debug(
                        f"Thread {tid}: Linking op '{op.name}' to CPU node before gap with rf_id "
                        f"'{last_cpu_node_rf_id}'."
                    )

            if last_cpu_node_rf_id:
                op.inter_thread_dep = last_cpu_node_rf_id

    def find_last_cpu_node_before_timestamp(
        self,
        ops_by_tid: Dict[int, List[KinetoOperator]],
        exclude_tid: int,
        timestamp: int,
    ) -> Optional[int]:
        """
        Find the last CPU node ID before a given timestamp in threads other than the excluded one.

        This ID is used to establish dependencies between groups across threads.

        Args:
            ops_by_tid (Dict[int, List[KinetoOperator]]): Operators grouped by thread ID.
            exclude_tid (int): Thread ID to exclude from the search.
            timestamp (int): Timestamp to compare against.

        Returns:
            Optional[int]: The ID of the last CPU node found, or None if not found.
        """
        logging.debug(f"Finding last CPU node before timestamp {timestamp} excluding thread {exclude_tid}.")
        last_cpu_node = None
        last_cpu_node_rf_id = None
        latest_timestamp = 0
        for tid, ops in ops_by_tid.items():
            if tid != exclude_tid:
                for op in sorted(ops, key=lambda op: op.timestamp):
                    if (
                        (op.category in ["cpu_op", "user_annotation"])
                        and (op.timestamp < timestamp)
                        and (op.timestamp > latest_timestamp)
                    ):
                        last_cpu_node = op
                        latest_timestamp = op.timestamp
                        last_cpu_node_rf_id = op.rf_id
        if last_cpu_node:
            logging.debug(f"Last CPU node before timestamp {timestamp} found: {last_cpu_node}")
        return last_cpu_node_rf_id

    def enforce_sync_dep(
        self,
        kineto_external_id_to_kineto_op_map: Dict[int, KinetoOperator],
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
        kineto_tid_ops_map: Dict[int, List[KinetoOperator]],
        sync_deps: Dict[int, List[int]],
    ):
        """
        Enforces synchronization order by storing Kineto ops that have synchronization dependency.

        Args:
            kineto_external_id_to_kineto_op_map (Dict[int, KinetoOperator]): Mapping between external ID and Kineto
                operators.
            sorted_kineto_cpu_ops (List[KinetoOperator]): Sorted list of Kineto CPU operators.
            sorted_kineto_cpu_op_ts (List[int]): Sorted list of timestamps for the Kineto CPU operators.
            kineto_tid_ops_map (Dict[int, List[KinetoOperator]]): Kineto operators grouped by thread ID.
            sync_deps (Dict[int, List[int]]): A dictionary mapping end event's external ID to a list of start event's
                external IDs that have synchronization dependencies.
        """
        logging.info("Enforcing sync order in Kineto traces.")

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.process_thread_sync_dep,
                    kineto_external_id_to_kineto_op_map,
                    sorted_kineto_cpu_ops,
                    sorted_kineto_cpu_op_ts,
                    tid,
                    ops,
                    sync_deps,
                ): tid
                for tid, ops in kineto_tid_ops_map.items()
            }

            for future in as_completed(futures):
                tid = futures[future]
                future.result()
                logging.debug(f"Thread {tid} sync dependencies processed.")

    def process_thread_sync_dep(
        self,
        kineto_external_id_to_kineto_op_map: Dict[int, KinetoOperator],
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
        tid: int,
        ops: List[KinetoOperator],
        sync_deps: Dict[int, List[int]],
    ) -> None:
        """
        Process synchronization dependencies for a specific thread.

        This method identifies synchronization dependencies for each operator within the current thread
        and updates the `sync_dep` attribute of each operator accordingly.

        Args:
            kineto_external_id_to_kineto_op_map (Dict[int, KinetoOperator]): Mapping between external ID and Kineto
                operators.
            sorted_kineto_cpu_ops (List[KinetoOperator]): Sorted list of Kineto CPU operators.
            sorted_kineto_cpu_op_ts (List[int]): Sorted list of timestamps for the Kineto CPU operators.
            tid (int): The current thread ID being processed.
            ops (List[KinetoOperator]): Kineto operators.
            sync_deps (Dict[int, List[int]]): A dictionary mapping end event's external ID to a list of start event's
                external IDs that have synchronization dependencies.
        """
        logging.info(f"Thread {tid}: Identifying synchronization dependency.")
        for op in ops:
            if op.external_id in sync_deps:
                sync_start_external_ids = sync_deps[op.external_id]

                for external_id in sync_start_external_ids:
                    if external_id in kineto_external_id_to_kineto_op_map:
                        start_sync_op = kineto_external_id_to_kineto_op_map[external_id]

                        # Find the closest Kineto operator with a start time later than the current op's timestamp
                        closest_start_kineto_op = self.find_closest_start_kineto_op(
                            op, sorted_kineto_cpu_ops, sorted_kineto_cpu_op_ts
                        )

                        # Add the external ID of the start_sync_op to closest_start_kineto_op.sync_dep if not present
                        if (closest_start_kineto_op is not None) and (
                            start_sync_op not in closest_start_kineto_op.sync_dep
                        ):
                            start_sync_op.sync_dep.append(closest_start_kineto_op)
                            logging.info(
                                f"Sync dependency: end op {closest_start_kineto_op.name} "
                                f"(external_id: {closest_start_kineto_op.external_id}, "
                                f"timestamp: {closest_start_kineto_op.timestamp})"
                                f" -> start op {start_sync_op.name} (external_id: {start_sync_op.external_id})"
                            )

    def find_closest_start_kineto_op(
        self, op: KinetoOperator, sorted_kineto_cpu_ops: List[KinetoOperator], sorted_kineto_cpu_op_ts: List[int]
    ) -> Optional[KinetoOperator]:
        """
        Find the closest start Kineto operator that occurs after the given operator's timestamp.

        Args:
            op (KinetoOperator): The current Kineto operator.
            sorted_kineto_cpu_ops (List[KinetoOperator]): Sorted list of Kineto CPU operators.
            sorted_kineto_cpu_op_ts (List[int]): Sorted list of timestamps for the Kineto CPU operators.

        Returns:
            Optional[KinetoOperator]: The closest start Kineto operator found, or None if not found.
        """
        index = bisect.bisect_right(sorted_kineto_cpu_op_ts, op.timestamp)
        closest_start_kineto_op = None

        for i in range(index, len(sorted_kineto_cpu_op_ts)):
            potential_sync_op = sorted_kineto_cpu_ops[i]
            if potential_sync_op.timestamp > op.timestamp:
                closest_start_kineto_op = potential_sync_op
                break

        return closest_start_kineto_op

    def link_traces(
        self,
        chakra_host_trace: str,
        host_ops: List[PyTorchOperator],
        kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
        kineto_correlation_cuda_runtime_map: Dict[int, KinetoOperator],
        kineto_rf_id_to_device_op_map: Dict[int, KinetoOperator],
        kineto_gpu_ops: List[KinetoOperator],
        kineto_thread_debug: Dict[int, Tuple[int, int]],
        kineto_process_start_time: int,
        kineto_process_end_time: int,
        kineto_external_id_to_kineto_op_map: Dict[int, KinetoOperator],
    ) -> Dict:
        """
        Link Chakra Host ET and Chakra Device ET to produce an enhanced Chakra ET (ET +).

        Args:
            chakra_host_trace (str): Path to the Chakra host execution trace file.
            host_ops (List[PyTorchOperator]): List of Chakra host operators.
            kineto_cpu_ops (List[KinetoOperator]): List of Kineto CPU operators.
            sorted_kineto_cpu_ops (List[KinetoOperator]): Sorted list of Kineto CPU operators.
            sorted_kineto_cpu_op_ts (List[int]): Sorted list of timestamps for the Kineto CPU operators.
            kineto_correlation_cuda_runtime_map (Dict[int, KinetoOperator]): Mapping between correlation IDs and
                kernel-launching CUDA runtime operators.
            kineto_rf_id_to_device_op_map (Dict[int, KinetoOperator]): Mapping between rf_id and Kineto operators.
            kineto_gpu_ops (List[KinetoOperator]): List of Kineto GPU operators.
            kineto_thread_debug (Dict[int, Tuple[int, int]]): debugrmation about threads, mapping thread IDs to a tuple
                of start and end times.
            kineto_process_start_time (int): Start time of the process, based on the earliest operator timestamp.
            kineto_process_end_time (int): End time of the process, based on the latest operator timestamp.
            kineto_external_id_to_kineto_op_map (Dict[int, KinetoOperator]): Mapping between external ID and Kineto
                operators.

        Returns:
            Dict: The enhanced Chakra Host Execution Trace (ET+).
        """
        logging.debug("Starting the process of linking Chakra host and device traces.")
        (
            kineto_cpu_ops,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
        ) = self.add_thread_and_process_annotations(
            kineto_cpu_ops,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
            kineto_thread_debug,
            kineto_process_start_time,
            kineto_process_end_time,
        )
        (
            host_op_id_to_kineto_ops_map,
            host_op_id_to_inclusive_dur_map,
            host_op_id_to_exclusive_dur_map,
            host_op_id_to_timestamp_map,
            host_op_id_to_inter_thread_dep_map,
        ) = self.map_host_to_device_ops(
            host_ops,
            kineto_cpu_ops,
            sorted_kineto_cpu_ops,
            sorted_kineto_cpu_op_ts,
            kineto_correlation_cuda_runtime_map,
            kineto_rf_id_to_device_op_map,
            kineto_gpu_ops,
            kineto_external_id_to_kineto_op_map,
        )
        chakra_execution_trace_plus_data = self.construct_et_plus_data(
            chakra_host_trace,
            host_op_id_to_kineto_ops_map,
            host_op_id_to_inclusive_dur_map,
            host_op_id_to_exclusive_dur_map,
            host_op_id_to_timestamp_map,
            host_op_id_to_inter_thread_dep_map,
        )
        logging.debug("Traces have been successfully linked.")
        return chakra_execution_trace_plus_data

    def add_thread_and_process_annotations(
        self,
        kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
        kineto_thread_debug: Dict[int, Tuple[int, int]],
        kineto_process_start_time: int,
        kineto_process_end_time: int,
    ) -> Tuple[List[KinetoOperator], List[KinetoOperator], List[int]]:
        """
        Add thread and process annotations to Kineto operators based on previously tracked timing debugrmation.

        These annotations are crucial for aligning Kineto operators with Chakra host nodes, ensuring completeness and
        compatibility of trace data for analysis. This method uses the process start and end times, as well as thread
        start and end times, collected during the categorization process to insert appropriate annotations directly
        into the Kineto operators list.
        """
        logging.debug("Adding process and thread annotations to Kineto operators.")

        # Insert process annotation operator. This operator represents the
        # overall time span of the trace process.
        process_annotation_op = KinetoOperator(
            {
                "name": EXECUTION_TRACE_PROCESS_ANNOTATION,
                "ts": kineto_process_start_time,
                "inclusive_dur": kineto_process_end_time - kineto_process_start_time,
                "exclusive_dur": 0,  # Process exclusive duration not applicable
            }
        )
        kineto_cpu_ops.insert(0, process_annotation_op)
        logging.debug(
            "Process annotation added with start time {} and duration {}.".format(
                kineto_process_start_time,
                kineto_process_end_time - kineto_process_start_time,
            )
        )

        # Insert thread annotation operators for each thread. These annotations
        # are crucial for understanding thread-level execution within the trace.
        for tid, (start_ts, end_ts) in kineto_thread_debug.items():
            inclusive_dur = end_ts - start_ts
            thread_annotation_op = KinetoOperator(
                {
                    "name": EXECUTION_TRACE_THREAD_ANNOTATION,
                    "ts": start_ts,
                    "inclusive_dur": inclusive_dur,
                    # Exclusive duration is set to zero in the final annotation. This is to avoid constraining
                    # the execution schedule to the original trace, allowing more flexibility in analyzing
                    # dependencies without being bound by specific execution timings.
                    "exclusive_dur": 0,
                }
            )
            # Find the correct position to insert the thread annotation
            position = next(
                (i for i, op in enumerate(kineto_cpu_ops) if op.tid == tid and op.timestamp >= start_ts),
                None,
            )
            if position is not None:
                kineto_cpu_ops.insert(position, thread_annotation_op)
            else:
                kineto_cpu_ops.append(thread_annotation_op)
            logging.debug(
                "Thread {} annotation added with start time {} and duration {}.".format(tid, start_ts, inclusive_dur)
            )

        sorted_kineto_cpu_ops = sorted(kineto_cpu_ops, key=lambda op: op.timestamp)
        sorted_kineto_cpu_op_ts = [op.timestamp for op in sorted_kineto_cpu_ops]

        return kineto_cpu_ops, sorted_kineto_cpu_ops, sorted_kineto_cpu_op_ts

    def map_host_to_device_ops(
        self,
        host_ops: List[PyTorchOperator],
        kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
        kineto_correlation_cuda_runtime_map: Dict[int, KinetoOperator],
        kineto_rf_id_to_device_op_map: Dict[int, KinetoOperator],
        kineto_gpu_ops: List[KinetoOperator],
        kineto_external_id_to_kineto_op_map,
    ) -> Tuple[
        Dict[int, List[KinetoOperator]],
        Dict[int, int],
        Dict[int, int],
        Dict[int, int],
        Dict[int, int],
    ]:
        """Map Chakra host operators to corresponding device operators."""
        logging.debug("Mapping Charka host operators to corresponding device operators.")
        cpu_external_id_to_gpu_ops_map = self.group_gpu_ops_by_cpu_launchers(
            kineto_gpu_ops, kineto_correlation_cuda_runtime_map, sorted_kineto_cpu_ops, sorted_kineto_cpu_op_ts
        )

        host_op_id_to_kineto_ops_map = {}
        host_op_id_to_inclusive_dur_map = {}
        host_op_id_to_exclusive_dur_map = {}
        host_op_id_to_timestamp_map = {}
        host_op_id_to_inter_thread_dep_map = {}

        for _, host_op in enumerate(host_ops):
            if (host_op.rf_id is not None) and (host_op.rf_id in kineto_rf_id_to_device_op_map):
                kineto_op = kineto_rf_id_to_device_op_map[host_op.rf_id]
                if kineto_op is None:
                    logging.warning(
                        f"No corresponding Kineto op found for Chakra host op ID: {host_op.id}, Name: "
                        f"'{host_op.name}'."
                    )
                    continue
                (
                    host_op_id_to_kineto_ops_map[host_op.id],
                    host_op_id_to_inclusive_dur_map[host_op.id],
                    host_op_id_to_exclusive_dur_map[host_op.id],
                    host_op_id_to_timestamp_map[host_op.id],
                    host_op_id_to_inter_thread_dep_map[host_op.id],
                ) = self.link_ops(
                    host_op,
                    kineto_op,
                    cpu_external_id_to_gpu_ops_map,
                    kineto_rf_id_to_device_op_map,
                    kineto_external_id_to_kineto_op_map,
                )

        logging.debug("Completed mapping of Chakra host operators to Kineto operators.")
        return (
            host_op_id_to_kineto_ops_map,
            host_op_id_to_inclusive_dur_map,
            host_op_id_to_exclusive_dur_map,
            host_op_id_to_timestamp_map,
            host_op_id_to_inter_thread_dep_map,
        )

    def group_gpu_ops_by_cpu_launchers(
        self,
        kineto_gpu_ops: List[KinetoOperator],
        kineto_correlation_cuda_runtime_map: Dict[int, KinetoOperator],
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
    ) -> Dict[int, List[KinetoOperator]]:
        """
        Group GPU operators based on their corresponding CPU launchers.

        This is determined by the 'external_id' which links GPU operators to their initiating CPU launcher events.

        Args:
            kineto_gpu_ops (List[KinetoOperator]): List of Kineto GPU operators.
            kineto_correlation_cuda_runtime_map (Dict[int, KinetoOperator]): Mapping between correlation IDs and
                kernel-launching CUDA runtime operators.
            sorted_kineto_cpu_ops (List[KinetoOperator]): Sorted list of Kineto CPU operators.
            sorted_kineto_cpu_op_ts (List[int]): Sorted list of timestamps extracted from Kineto operators for
                efficient temporal queries.

        Returns:
            Dict[int, List[KinetoOperator]]: Mapping from CPU launch event indices to GPU operators.

        Raises:
            ValueError: If 'external_id' is missing for any GPU operator.
        """
        cpu_external_id_to_gpu_ops_map = {}
        for gpu_op in kineto_gpu_ops:
            parent_cpu_op = self.find_parent_cpu_op(
                gpu_op, kineto_correlation_cuda_runtime_map, sorted_kineto_cpu_ops, sorted_kineto_cpu_op_ts
            )
            if not parent_cpu_op:
                warning_msg = f"Missing parent CPU operator for GPU op '{gpu_op.name}'. Orphaned GPU operator."
                logging.warning(warning_msg)
                continue

            if parent_cpu_op.external_id == "":
                error_msg = (
                    f"Missing 'external_id' for CPU operator {parent_cpu_op.name}. "
                    f"Cannot link GPU op {gpu_op.name} to {parent_cpu_op.name}."
                )
                logging.warning(error_msg)
                continue

            logging.debug(f"group_gpu_ops_by_cpu_launchers '{parent_cpu_op.name}' -> '{gpu_op.name}'")

            if "ncclDevKernel_SendRecv" in gpu_op.name:
                if parent_cpu_op.dst_rank is not None:
                    gpu_op.dst_rank = parent_cpu_op.dst_rank
                if parent_cpu_op.src_rank is not None:
                    gpu_op.src_rank = parent_cpu_op.src_rank

            cpu_external_id_to_gpu_ops_map.setdefault(parent_cpu_op.external_id, []).append(gpu_op)

        return cpu_external_id_to_gpu_ops_map

    def find_parent_cpu_op(
        self,
        kineto_gpu_op: KinetoOperator,
        kineto_correlation_cuda_runtime_map: Dict[int, KinetoOperator],
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
    ) -> Optional[KinetoOperator]:
        """
        Find the parent CPU operator for a given GPU operator by identifying the corresponding CUDA runtime operator.

        It then locates the closest preceding CPU operator based on the CUDA runtime's timestamp, considering the
        temporal distance between the GPU operation's start and the initiating CPU operation.

        Args:
            kineto_gpu_op (KinetoOperator): The GPU operator.
            kineto_correlation_cuda_runtime_map (Dict[int, KinetoOperator]): Mapping between correlation IDs and
                kernel-launching CUDA runtime operators.
            sorted_kineto_cpu_ops (List[KinetoOperator]): Sorted list of Kineto CPU operators.
            sorted_kineto_cpu_op_ts (List[int]): Sorted list of timestamps extracted from Kineto operators for
                efficient temporal queries.

        Returns:
            Optional[KinetoOperator]: The parent CPU operator if found.

        Raises:
            ValueError: If no CUDA runtime operator is found for the given correlation ID.
        """
        if kineto_gpu_op.correlation not in kineto_correlation_cuda_runtime_map:
            warning_msg = (
                f"No CUDA runtime operator found for correlation ID {kineto_gpu_op.correlation}. "
                "This is not a common case, and there should be a corresponding CUDA runtime operator for a given GPU "
                "kernel operator. It can be a case where CUDA runtime operators are not properly identified and added "
                "to the map, kineto_correlation_cuda_runtime_map. Please manually check if the corresponding CUDA "
                "runtime operator with the correlation is dropped by mistake. It is likely that it is because of "
                "incomplete map, cuda_launch_operations, in is_kernel_launch_op. Please update the map properly to "
                "cover all CUDA runtime launch operators."
            )
            logging.warning(warning_msg)
            return None

        kineto_runtime_op = kineto_correlation_cuda_runtime_map[kineto_gpu_op.correlation]
        kineto_gpu_op.tid = kineto_runtime_op.tid
        logging.debug(
            f"Found CUDA runtime operation '{kineto_runtime_op.name}' for GPU operator '{kineto_gpu_op.name}'."
        )

        # Find the closest CPU operator that precedes the CUDA runtime operation
        parent_cpu_op = self.find_closest_op(
            kineto_gpu_op, sorted_kineto_cpu_ops, sorted_kineto_cpu_op_ts, kineto_runtime_op.timestamp
        )
        if not parent_cpu_op:
            logging.warning(
                f"No parent CPU operator found for GPU operator '{kineto_gpu_op.name}' "
                f"linked to CUDA runtime operation '{kineto_runtime_op.name}' "
                f"(ts: {kineto_runtime_op.timestamp})."
            )

        return parent_cpu_op

    def find_closest_op(
        self,
        kineto_gpu_op: KinetoOperator,
        sorted_kineto_cpu_ops: List[KinetoOperator],
        sorted_kineto_cpu_op_ts: List[int],
        ts: int,
    ) -> Optional[KinetoOperator]:
        """
        Find the Kineto operator that is closest in start time to a given timestamp and that covers the timestamp.

        Args:
            kineto_gpu_op (KinetoOperator): The GPU operator being compared.
            sorted_kineto_cpu_ops (List[KinetoOperator]): List of Kineto operators.
            sorted_kineto_cpu_op_ts (List[int]): List of timestamps for the sorted Kineto operators.
            ts (int): The timestamp to compare against.

        Returns:
            Optional[KinetoOperator]: The closest Kineto operator if found.
        """
        # Step 1: Find the initial closest index
        index = bisect.bisect_left(sorted_kineto_cpu_op_ts, ts)

        if index == 0:
            # All operators are later than the timestamp
            return None

        # Step 2: Find the closest operator
        tid_only_match = None  # Track the best operator with matching tid
        for i in range(index - 1, -1, -1):
            op = sorted_kineto_cpu_ops[i]
            # Skip 'nccl:coalesced' for NCCL-related GPU operations
            if "nccl" in kineto_gpu_op.name.lower() and op.name == "nccl:coalesced":
                continue
            # Return the operator matching both tid and external_id
            if op.tid == kineto_gpu_op.tid and op.external_id == kineto_gpu_op.external_id:
                return op
            # Track the tid_only_match operator with matching tid if no full match is found
            if tid_only_match is None and op.tid == kineto_gpu_op.tid:
                tid_only_match = op

        # Step 3: Return the best match or None if no match is found
        return tid_only_match

    def link_ops(
        self,
        host_op: PyTorchOperator,
        kineto_op: KinetoOperator,
        cpu_external_id_to_gpu_ops_map: Dict[int, List[KinetoOperator]],
        kineto_rf_id_to_device_op_map: Dict[int, KinetoOperator],
        kineto_external_id_to_kineto_op_map: Dict[int, KinetoOperator],
    ) -> Tuple[List[KinetoOperator], int, int, int, Optional[int]]:
        """
        Link a Chakra host operator to its corresponding Kineto operator and any associated GPU operators.

        Args:
            host_op (PyTorchOperator): Chakra host operator to link.
            kineto_op (KinetoOperator): Corresponding Kineto operator.
            cpu_external_id_to_gpu_ops_map (Dict[int, List[KinetoOperator]]): GPU ops mapping.
            kineto_rf_id_to_device_op_map (Dict[int, KinetoOperator]): Kineto operator mapping.
            kineto_external_id_to_kineto_op_map (Dict[int, KinetoOperator]): Mapping from external id to
                KinetoOperators.

        Returns:
            Tuple containing:
                - List[KinetoOperator]: The list of linked Kineto GPU operators.
                - int: The inclusive duration of the linked Kineto operator.
                - int: The exclusive duration of the linked Kineto operator.
                - int: The timestamp of the linked Kineto operator.
                - Optional[int]: The inter-thread dependency ID if present.
                - List[int]: List of synchronization dependency IDs.
        """
        kineto_op.host_op = host_op
        linked_gpu_ops = cpu_external_id_to_gpu_ops_map.get(kineto_op.external_id, [])
        inclusive_dur = kineto_op.inclusive_dur
        exclusive_dur = kineto_op.exclusive_dur
        timestamp = kineto_op.timestamp

        inter_thread_dep = self.get_inter_thread_dep(kineto_op, kineto_rf_id_to_device_op_map)

        self.link_gpu_ops(host_op, linked_gpu_ops)

        return linked_gpu_ops, inclusive_dur, exclusive_dur, timestamp, inter_thread_dep

    def get_inter_thread_dep(self, kineto_op, kineto_rf_id_to_device_op_map):
        """
        Retrieve the inter-thread dependency ID for a given Kineto operator.

        This method finds the corresponding Chakra host operator ID for the inter-thread dependency if it exists.

        Args:
            kineto_op (KinetoOperator): The Kineto operator being processed.
            kineto_rf_id_to_device_op_map (Dict[int, KinetoOperator]): Mapping from rf_id to Kineto operators.

        Returns:
            Optional[int]: The Chakra host operator ID for the inter-thread dependency if it exists, otherwise None.
        """
        if kineto_op.inter_thread_dep:
            inter_thread_dep_kineto_op = kineto_rf_id_to_device_op_map[kineto_op.inter_thread_dep]
            if inter_thread_dep_kineto_op.host_op:
                return inter_thread_dep_kineto_op.host_op.id
        return None

    def link_gpu_ops(self, host_op: PyTorchOperator, kineto_gpu_ops: List[KinetoOperator]) -> None:
        """
        Link GPU operators to a Chakra host operator.

        Args:
            host_op (PyTorchOperator): The Chakra host operator to link to.
            kineto_gpu_ops (List[KinetoOperator]): GPU operators to link.
        """
        for gpu_op in kineto_gpu_ops:
            gpu_op.parent_host_op_id = host_op.id

    def construct_et_plus_data(
        self,
        chakra_host_trace: str,
        host_op_id_to_kineto_ops_map: Dict[int, List[KinetoOperator]],
        host_op_id_to_inclusive_dur_map: Dict[int, int],
        host_op_id_to_exclusive_dur_map: Dict[int, int],
        host_op_id_to_timestamp_map: Dict[int, int],
        host_op_id_to_inter_thread_dep_map: Dict[int, int],
    ) -> Dict:
        """
        Construct the enhanced Chakra Host Execution Trace (ET+) data structure.

        This method enriches the Chakra host execution trace with detailed performance data from the Kineto trace,
        offering a comprehensive view of the execution.

        Args:
            chakra_host_trace (str): Path to the Chakra host execution trace file.
            host_op_id_to_kineto_ops_map (Dict[int, List[KinetoOperator]]): Map from Chakra host op IDs to Kineto
                GPU ops.
            host_op_id_to_inclusive_dur_map (Dict[int, int]): Inclusive duration map for Chakra host ops.
            host_op_id_to_exclusive_dur_map (Dict[int, int]): Exclusive duration map for Chakra host ops.
            host_op_id_to_timestamp_map (Dict[int, int]): Timestamp map for Chakra host ops.
            host_op_id_to_inter_thread_dep_map (Dict[int, int]): Mapping of Chakra host operator IDs to IDs of
                latest CPU node from other threads before the gap.

        Returns:
            Dict: The constructed ET+ data.
        """
        logging.debug("Constructing ET+ data.")
        with open(chakra_host_trace, "r") as file:
            pytorch_et_data = json.load(file)

        sorted_nodes = sorted(pytorch_et_data["nodes"], key=lambda x: x["id"])
        gpu_ops = []
        for op in sorted_nodes:
            gpu_ops += self.process_op_and_dependents(
                op,
                host_op_id_to_kineto_ops_map,
                host_op_id_to_inclusive_dur_map,
                host_op_id_to_exclusive_dur_map,
                host_op_id_to_timestamp_map,
                host_op_id_to_inter_thread_dep_map,
            )
        pytorch_et_data["nodes"] += gpu_ops

        # Add sync dependencies
        sync_dep_mapping = {}
        for gpu_op in gpu_ops:
            if "sync_dep_to" in gpu_op:
                for sync_dep_to in gpu_op["sync_dep_to"]:
                    if sync_dep_to not in sync_dep_mapping:
                        sync_dep_mapping[sync_dep_to] = []
                    sync_dep_mapping[sync_dep_to].append(gpu_op["id"])
                del gpu_op["sync_dep_to"]

        # Update parent-child relationships with new IDs
        sorted_nodes = sorted(pytorch_et_data["nodes"], key=lambda x: x["id"])
        for op in sorted_nodes:
            for key in sync_dep_mapping:
                if self.id_assigner.lookup_new_id(key) == op["id"]:
                    op["sync_dep"] = sync_dep_mapping[key]
            if "ctrl_deps" in op:
                op["ctrl_deps"] = self.id_assigner.assign_or_retrieve_id(op["ctrl_deps"])

        return pytorch_et_data

    def process_op_and_dependents(
        self,
        op: Dict,
        host_op_id_to_kineto_ops_map: Dict[int, List[KinetoOperator]],
        host_op_id_to_inclusive_dur_map: Dict[int, int],
        host_op_id_to_exclusive_dur_map: Dict[int, int],
        host_op_id_to_timestamp_map: Dict[int, int],
        host_op_id_to_inter_thread_dep_map: Dict[int, int],
    ) -> List[Dict]:
        """
        Process a single operator in the Chakra host trace, assign a unique ID, and process any dependent operators.

        Args:
            op (Dict): The operator to be processed.
            host_op_id_to_kineto_ops_map (Dict[int, List[KinetoOperator]]): Map from Chakra host op IDs to Kineto GPU
                ops.
            host_op_id_to_inclusive_dur_map (Dict[int, int]): Inclusive duration map for Chakra host ops.
            host_op_id_to_exclusive_dur_map (Dict[int, int]): Exclusive duration map for Chakra host ops.
            host_op_id_to_timestamp_map (Dict[int, int]): Timestamp map for Chakra host ops.
            host_op_id_to_inter_thread_dep_map (Dict[int, int]): Mapping of Chakra host operator IDs to IDs of latest
                CPU node from other threads before the gap.

        Returns:
            List[Dict]: A list of GPU operators processed and linked to the given operator.
        """
        orig_op_id = op["id"]
        new_op_id = self.id_assigner.assign_or_retrieve_id(orig_op_id)
        op["id"] = new_op_id

        # Update operator with Kineto data if available
        if orig_op_id in host_op_id_to_inclusive_dur_map:
            op["inclusive_dur"] = host_op_id_to_inclusive_dur_map[orig_op_id]
            op["exclusive_dur"] = host_op_id_to_exclusive_dur_map[orig_op_id]
            op["ts"] = host_op_id_to_timestamp_map[orig_op_id]
            if orig_op_id in host_op_id_to_inter_thread_dep_map:
                op["inter_thread_dep"] = self.id_assigner.lookup_new_id(host_op_id_to_inter_thread_dep_map[orig_op_id])
            else:
                op["inter_thread_dep"] = None

        # Process and append dependent GPU operators
        if orig_op_id in host_op_id_to_kineto_ops_map:
            gpu_ops = self.process_dependent_gpu_ops(op, orig_op_id, host_op_id_to_kineto_ops_map)
            host_op_id_to_kineto_ops_map.pop(orig_op_id)
            return gpu_ops
        return []

    def process_dependent_gpu_ops(
        self, cpu_op: Dict, orig_op_id: int, host_op_id_to_kineto_ops_map: Dict[int, List[KinetoOperator]]
    ) -> List[Dict]:
        """
        Create and return a list of GPU operators that are dependent on a specific CPU operator.

        The GPU operators are deep copies of the existing operators with updated IDs and other relevant
        fields from the CPU operator.

        Args:
            cpu_op (Dict): The Chakra host CPU operator.
            orig_op_id (int): The original ID of the CPU operator.
            host_op_id_to_kineto_ops_map (Dict[int, List[KinetoOperator]]): Map from host operator IDs to device
                operators

        Returns:
            List[Dict]: A list of processed GPU operators.
        """
        updated_gpu_ops = []
        dependent_gpu_ops = host_op_id_to_kineto_ops_map.get(orig_op_id, [])
        for gpu_op in sorted(dependent_gpu_ops, key=lambda x: x.timestamp):
            new_gpu_op = copy.deepcopy(cpu_op)
            new_gpu_op_id = self.id_assigner.generate_new_id()
            new_gpu_op.update(
                {
                    "id": new_gpu_op_id,
                    "ctrl_deps": orig_op_id,
                    "inputs": cpu_op["inputs"],
                    "outputs": cpu_op["outputs"],
                    "cat": gpu_op.category,
                    "name": gpu_op.name,
                    "ph": gpu_op.phase,
                    "inclusive_dur": gpu_op.inclusive_dur,
                    "exclusive_dur": gpu_op.exclusive_dur,
                    "ts": gpu_op.timestamp,
                    "stream": gpu_op.stream,
                    **(
                        {"pg_name": gpu_op.pg_name}
                        if gpu_op.is_inter_gpu_comms_op() and gpu_op.pg_name is not None
                        else {}
                    ),
                    **(
                        {"dst_rank": gpu_op.dst_rank}
                        if "ncclDevKernel_SendRecv" in gpu_op.name and gpu_op.dst_rank is not None
                        else {}
                    ),
                    **(
                        {"src_rank": gpu_op.src_rank}
                        if "ncclDevKernel_SendRecv" in gpu_op.name and gpu_op.src_rank is not None
                        else {}
                    ),
                }
            )
            updated_gpu_ops.append(new_gpu_op)

            for sync_dep in gpu_op.sync_dep:
                if sync_dep.host_op:
                    if "sync_dep_to" not in new_gpu_op:
                        new_gpu_op["sync_dep_to"] = []
                    if self.id_assigner.lookup_new_id(sync_dep.host_op.id) not in new_gpu_op["sync_dep_to"]:
                        new_gpu_op["sync_dep_to"].append(self.id_assigner.lookup_new_id(sync_dep.host_op.id))

        return updated_gpu_ops

    def dump_chakra_execution_trace_plus(self, chakra_execution_trace_plus_data: Dict, output_file: str) -> None:
        """
        Dump the enhanced Chakra execution trace plus data to a file.

        Args:
            chakra_execution_trace_plus_data (Dict): The constructed ET+ data.
            output_file (str): The file path where the ET+ data will be saved.
        """
        logging.debug(f"Starting to dump ET+ data to {output_file}.")

        if chakra_execution_trace_plus_data is None:
            logging.error("ET+ data not constructed. Please run construct_et_plus_data first.")
            return

        if "nodes" in chakra_execution_trace_plus_data:
            chakra_execution_trace_plus_data["nodes"] = sorted(
                chakra_execution_trace_plus_data["nodes"], key=lambda x: x["id"]
            )

        with open(output_file, "w") as file:
            json.dump(chakra_execution_trace_plus_data, file, indent=4)
        logging.debug(f"ET+ data dumped to {output_file}.")
