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

  try {
    readGlobalMetadata();
    readNextWindow();
  } catch (const std::exception& e) {
    cerr << "Error in constructor: " << e.what() << endl;
    throw; // Rethrow the exception for caller to handle
  }
}

ETFeeder::~ETFeeder() {}

void ETFeeder::addNode(shared_ptr<ETFeederNode> node) {
  dep_graph_[node->getChakraNode()->id()] = node;
  auto node_id = node->getChakraNode()->id();
  if (node->getChakraNode()->data_deps().size() == 0){
    // std::cout << "Adding node " << node_id << " as a parentless node " << std::endl;
    if (node->is_cpu_op()) {
      dep_resolved_nodes_[CPU_QUEUE].push(node);
    } else {
      dep_resolved_nodes_[GPU_QUEUE].push(node);
    }
  }
}

void ETFeeder::removeNode(uint64_t node_id) {
  dep_graph_.erase(node_id);

  if (!et_complete_ && (dep_free_node_queue_.size() < window_size_)) {
    readNextWindow();
  }
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
  shared_ptr<ETFeederNode> node = dep_graph_[node_id];
  for (auto child : node->getChildren()) {
    auto child_chakra = child->getChakraNode();
    for (auto it = child_chakra->mutable_data_deps()->begin();
         it != child_chakra->mutable_data_deps()->end();
         ++it) {
      if (*it == node_id) {
        child_chakra->mutable_data_deps()->erase(it);
        break;
      }
    }
    if (child_chakra->data_deps().size() == 0) {
      auto child_node_id = child->id();
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
