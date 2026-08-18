#include "et_feeder.h"

#include <iostream>

using namespace std;
using namespace Chakra;

ETFeeder::ETFeeder(string filename)
    : trace_(filename), window_size_(4096 * 256), et_complete_(false) {
  if (!trace_.is_open()) { // Assuming a method to check if file is open
    throw std::runtime_error("Failed to open trace file: " + filename);
  }
  dep_resolved_nodes_[CPU_QUEUE] = {};
  dep_resolved_nodes_[GPU_QUEUE] = {};
  initial_dep_resolved_nodes_[CPU_QUEUE] = {};
  initial_dep_resolved_nodes_[GPU_QUEUE] = {};

  try {
    readGlobalMetadata();
    readNextWindow();
    initial_dep_graph_ = dep_graph_;
  } catch (const std::exception& e) {
    cerr << "Error in constructor: " << e.what() << endl;
    throw; // Rethrow the exception for caller to handle
  }
}

ETFeeder::~ETFeeder() {}

void ETFeeder::addNode(shared_ptr<ETFeederNode> node) {
  auto node_id = node->getChakraNode()->id();
  dep_graph_[node_id] = node;
  uint32_t num_deps = node->getChakraNode()->data_deps().size();
  initial_dep_count_[node_id] = num_deps;
  remaining_dep_count_[node_id] = num_deps;
  if (num_deps == 0){
    // std::cout << "Adding node " << node_id << " as a parentless node " << std::endl;
    DepQueue queue_idx = node->is_cpu_op() ? CPU_QUEUE : GPU_QUEUE;
    dep_resolved_nodes_[queue_idx].push(node);
    // Also record this node in the read-only snapshot of initially
    // dep-free nodes, so resetIteration() can restore dep_resolved_nodes_
    // without re-scanning the whole dep_graph_.
    initial_dep_resolved_nodes_[queue_idx].push(node);
  }
}

void ETFeeder::removeNode(uint64_t node_id) {
  dep_graph_.erase(node_id);

  if (!et_complete_ && (dep_free_node_queue_.size() < window_size_)) {
    readNextWindow();
  }
}

void ETFeeder::resetIteration() {
  // Restore the per-iteration map from the pristine, read-only source. Both
  // maps hold the same underlying shared_ptr<ETFeederNode> objects, but since
  // those objects are no longer mutated (see freeChildrenNodes()), this is
  // safe to reuse across iterations.
  dep_graph_ = initial_dep_graph_;
  // Flush the per-iteration dependency-resolution state back to its initial
  // (read-only) values.
  remaining_dep_count_ = initial_dep_count_;
  // Restore dep_resolved_nodes_ from the read-only snapshot taken when the
  // trace was first loaded, instead of re-deriving it by scanning every node
  // in dep_graph_ and checking remaining_dep_count_ == 0.
  dep_resolved_nodes_ = initial_dep_resolved_nodes_;
  // Note: et_complete_ intentionally left untouched. It reflects whether the
  // underlying trace file stream has been fully consumed, which is a
  // property of the file, not of a single iteration, and must not be reset.
}

bool ETFeeder::hasNodesToIssue() {
  return !(dep_graph_.empty() && dep_resolved_nodes_[CPU_QUEUE].empty() && dep_resolved_nodes_[GPU_QUEUE].empty());
}

shared_ptr<ETFeederNode> ETFeeder::getNextIssuableNode(DepQueue which_queue) {
  if (dep_resolved_nodes_[which_queue].size() != 0) {
    auto old_size = dep_resolved_nodes_[which_queue].size();
    shared_ptr<ETFeederNode> node = dep_resolved_nodes_[which_queue].front();
    dep_resolved_nodes_[which_queue].pop();
    auto popped_node_id = node->id();
    auto new_size = dep_resolved_nodes_[which_queue].size();
    // std::cout << "Pop node " << node->id() << " from " << which_queue << " queue. Old size: " << old_size << ", New size: " << new_size << std::endl;
    return node;
  } else {
    return nullptr;
  }
}

shared_ptr<ETFeederNode> ETFeeder::lookupNode(uint64_t node_id) {
  try {
    return dep_graph_.at(node_id);
  } catch (const std::out_of_range& e) {
    std::cerr << "looking for node_id=" << node_id
              << " in dep graph, however, not loaded yet" << std::endl;
    throw(e);
  }
}

void ETFeeder::freeChildrenNodes(uint64_t node_id) {
  // NOTE: This intentionally does NOT mutate any ETFeederNode/proto state
  // (e.g. via mutable_data_deps()->erase(...)). Those objects are shared
  // between initial_dep_graph_ and dep_graph_ (a shallow map copy), so
  // mutating them would permanently corrupt the read-only initial graph and
  // break subsequent iterations. Instead, track remaining dependency counts
  // in a separate per-iteration map that gets reset in resetIteration().
  shared_ptr<ETFeederNode> node = dep_graph_[node_id];
  for (auto child : node->getChildren()) {
    auto child_node_id = child->id();
    auto count_it = remaining_dep_count_.find(child_node_id);
    if (count_it == remaining_dep_count_.end() || count_it->second == 0) {
      continue;
    }
    --(count_it->second);
    if (count_it->second == 0) {
      DepQueue child_queue_idx = UNKNOWN_VALUE;
      if (child->is_cpu_op()) {
        child_queue_idx = CPU_QUEUE;
      } else {
        child_queue_idx = GPU_QUEUE;
      }
      dep_resolved_nodes_[child_queue_idx].push(child);
    }
  }
}

void ETFeeder::readGlobalMetadata() {
  if (!trace_.is_open()) {
    throw runtime_error(
        "Trace file closed unexpectedly during reading global metadata.");
  }
  shared_ptr<ChakraProtoMsg::GlobalMetadata> pkt_msg =
      make_shared<ChakraProtoMsg::GlobalMetadata>();
  trace_.read(*pkt_msg);
}

shared_ptr<ETFeederNode> ETFeeder::readNode() {
  shared_ptr<ChakraProtoMsg::Node> pkt_msg =
      make_shared<ChakraProtoMsg::Node>();
  if (!trace_.read(*pkt_msg)) {
    return nullptr;
  }
  shared_ptr<ETFeederNode> node = make_shared<ETFeederNode>(pkt_msg);

  bool dep_unresolved = false;
  for (int i = 0; i < pkt_msg->data_deps_size(); ++i) {
    auto parent_node = dep_graph_.find(pkt_msg->data_deps(i));
    if (parent_node != dep_graph_.end()) {
      parent_node->second->addChild(node);
    } else {
      dep_unresolved = true;
      node->addDepUnresolvedParentID(pkt_msg->data_deps(i));
    }
  }

  return node;
}

void ETFeeder::readNextWindow() {
  if (!trace_.is_open()) {
    throw runtime_error(
        "Trace file closed unexpectedly during reading next window.");
  }
  uint32_t num_read = 0;
  do {
    shared_ptr<ETFeederNode> new_node = readNode();
    if (new_node == nullptr) {
      et_complete_ = true;
      break;
    }

    addNode(new_node);
    ++num_read;

  } while ((num_read < window_size_));
}
