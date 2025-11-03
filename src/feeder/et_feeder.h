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

enum DepQueue {
  UNKNOWN_VALUE,
  CPU_QUEUE,
  GPU_QUEUE
};

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

  std::unordered_map<uint64_t, std::shared_ptr<ETFeederNode>> dep_graph_{};
  // A map that goes from "which queue" to a queue of ET Nodes that have all deps. resolved and ready to launch.
  std::unordered_map<DepQueue, std::queue<std::shared_ptr<ETFeederNode>>> dep_resolved_nodes_{};
};

} // namespace Chakra
