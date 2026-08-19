#pragma once

#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "et_feeder_node.h"
#include "protoio.hh"

namespace Chakra {
struct CompareNodes : public std::binary_function<
                          std::shared_ptr<ETFeederNode>,
                          std::shared_ptr<ETFeederNode>,
                          bool> {
  bool operator()(
      const std::shared_ptr<ETFeederNode> lhs,
      const std::shared_ptr<ETFeederNode> rhs) const {
    return lhs->getChakraNode()->id() > rhs->getChakraNode()->id();
  }
};

// enum DepQueue {
//   UNKNOWN_VALUE,
//   CPU_QUEUE,
//   GPU_QUEUE
// };

using DepQueue = int64_t;

class ETFeeder {
 public:
  ETFeeder(std::string filename);
  ~ETFeeder();

  void addNode(std::shared_ptr<ETFeederNode> node);
  void removeNode(uint64_t node_id);
  bool hasNodesToIssue();
  std::shared_ptr<ETFeederNode> getNextIssuableNode(DepQueue which_queue);
  // void pushBackIssuableNode(uint64_t node_id);
  std::shared_ptr<ETFeederNode> lookupNode(uint64_t node_id);
  void freeChildrenNodes(uint64_t node_id);
  const std::unordered_set<DepQueue>& getSeenTids() const;
  void resetIteration();

  std::priority_queue<
      std::shared_ptr<ETFeederNode>,
      std::vector<std::shared_ptr<ETFeederNode>>,
      CompareNodes>
      dep_free_node_queue_{};
 private:
  void readGlobalMetadata();
  std::shared_ptr<ETFeederNode> readNode();
  void readNextWindow();
  void resolveDep();

  ProtoInputStream trace_;
  const uint32_t window_size_;
  bool et_complete_;

  // Read-only, populated once when the trace is first loaded. Never mutated
  // afterwards so it can be reused as the pristine source of truth for every
  // iteration.
  std::unordered_map<uint64_t, std::shared_ptr<ETFeederNode>> initial_dep_graph_{};
  // Per-iteration map. Entries are erased via removeNode() as nodes complete
  // and repopulated from initial_dep_graph_ at the start of each iteration.
  std::unordered_map<uint64_t, std::shared_ptr<ETFeederNode>> dep_graph_{};
  // A map that goes from "which queue" to a queue of ET Nodes that have all deps. resolved and ready to launch.
  std::unordered_map<DepQueue, std::queue<std::shared_ptr<ETFeederNode>>> dep_resolved_nodes_{};
  // Read-only snapshot of dep_resolved_nodes_ as it looked right after the
  // trace was first loaded (i.e. the set of parentless/dep-free nodes).
  // Populated once in addNode() and never mutated afterwards, so
  // resetIteration() can restore dep_resolved_nodes_ via direct assignment
  // instead of re-scanning the entire dep_graph_ every iteration.
  std::unordered_map<DepQueue, std::queue<std::shared_ptr<ETFeederNode>>> initial_dep_resolved_nodes_{};
  std::unordered_set<DepQueue> seen_tids_{};

  // Read-only per-node count of unresolved data_deps, as seen when the node
  // was first loaded from the trace. Never mutated after being set, so it
  // can be reused across iterations without touching the shared, immutable
  // ETFeederNode/proto objects in initial_dep_graph_.
  std::unordered_map<uint64_t, uint32_t> initial_dep_count_{};
  // Per-iteration, mutable count of remaining unresolved data_deps for each
  // node. Decremented in freeChildrenNodes() as parents complete, and reset
  // from initial_dep_count_ at the start of each iteration.
  std::unordered_map<uint64_t, uint32_t> remaining_dep_count_{};
};

} // namespace Chakra
