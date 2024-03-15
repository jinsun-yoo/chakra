#!/usr/bin/env python3

import logging

from  xml.etree import ElementTree
from typing import Any, List
from chakra.third_party.utils.protolib import encodeMessage as encode_message
from chakra.et_def.et_def_pb2 import (
    NodeType,
    Node,
    AttributeProto as ChakraAttr,
    COMP_NODE,
    COMM_COLL_NODE,
    ALL_REDUCE,
    ALL_TO_ALL,
    ALL_GATHER,
    REDUCE_SCATTER,
    COMM_SEND_NODE,
    COMM_RECV_NODE,
    GlobalMetadata
)

HARDCODE_COMM_SIZE = int(1024 * 1024 / 4) # Bytes
HARDCODE_LOCAL_BW = 50
# 1000 b/c microsecond to nanosecond. Refer to Workload::issue_replay
HARDCODE_COMP_TIME_NS = int (3 * int(HARDCODE_COMM_SIZE / HARDCODE_LOCAL_BW) / 1000)

class MSCCL2ChakraConverter:
    def __init__(
        self,
        input_filename: str,
        output_filename: str,
        logger: logging.Logger
    ) -> None:
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.logger = logger
        self.next_node_id = 0

    # Creates the global metadata info that is added to the start of all ET files.
    def create_global_metadata(self):
        input_text = ""
        with open(self.input_filename, "r") as input_file:
            input_text = input_file.read()
        attr = [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="input_file", string_val=input_text)
        ]
        metadata = GlobalMetadata(attr=attr)
        return metadata

    # Creates an ET node, and assigns a node id to it.
    # Increment the node id, to be assigned to the next ET node.
    def create_et_node(
        self,
        name: str,
        node_type: NodeType
    ) -> Any:
        node = Node()
        node.id = self.next_node_id
        self.next_node_id += 1
        node.name = name
        node.type = node_type
        return node

    # This function is called to reset the node id when starting to add nodes for a new ET trace file.
    # There will be one ET trace file for each NPU. 
    def reset_node_id(
            self
    ): self.next_node_id = 0

    def get_comp_node(
        self,
        tb_xml_node: ElementTree.Element,
        step_id: int
    ) -> Any:
        tb_id = tb_xml_node.attrib['id']
        node = self.create_et_node(f"COMP_NODE_tb{tb_id}_step{step_id}",
                             COMP_NODE)
        node.duration_micros = HARDCODE_COMP_TIME_NS
        return node

    def get_send_node (
        self, 
        tb_xml_node: ElementTree.Element,
        step_id: int
    ) -> Any:
        tb_id = tb_xml_node.attrib['id']
        layer_name = f'COMM_SEND_NODE_tb{tb_id}_step{step_id}'
        dst = int(tb_xml_node.attrib['send'])
        tag = int(tb_xml_node.attrib['chan'])
        size = HARDCODE_COMM_SIZE

        node = self.create_et_node(layer_name, COMM_SEND_NODE)
        node.attr.append(ChakraAttr(name="comm_type",
                                    int64_val=COMM_SEND_NODE))
        node.attr.append(ChakraAttr(name="comm_size",
                                    uint64_val=size))
        node.attr.append(ChakraAttr(name="comm_dst",
                                    uint64_val=dst))
        node.attr.append(ChakraAttr(name="comm_tag",
                                    uint64_val=tag))
        return node


    def get_recv_node (
        self, 
        tb_xml_node: ElementTree.Element,
        step_id: int
    ) -> Any:
        tb_id = tb_xml_node.attrib['id']
        layer_name = f'COMM_RECV_NODE_tb{tb_id}_step{step_id}'
        src = int(tb_xml_node.attrib['recv'])
        tag = int(tb_xml_node.attrib['chan'])
        size = HARDCODE_COMM_SIZE

        node = self.create_et_node(layer_name, COMM_RECV_NODE)
        node.attr.append(ChakraAttr(name="comm_type",
                                    int64_val=COMM_RECV_NODE))
        node.attr.append(ChakraAttr(name="comm_size",
                                    uint64_val=size))
        node.attr.append(ChakraAttr(name="comm_src",
                                    uint64_val=src))
        node.attr.append(ChakraAttr(name="comm_tag",
                                    uint64_val=tag))
        return node

    # Add 'parent_node' as the parent to 'child_node'.
    # Note that parent_node and child_node has to be within the same ET trace file.
    def add_parent(
        self,
        child_node: Any,
        parent_node: Any
    ) -> None:
        child_node.data_deps.append(parent_node.id)
        
    def convert(self) -> None:
        node_map = {}
        step_map = {}
        tree = ElementTree.parse(self.input_filename)
        root = tree.getroot()

        # Read the XML file and create ET Trace nodes. 
        for gpu in root.findall('gpu'):
            gpu_id = int(gpu.attrib['id'])
            node_map[gpu_id] = {}
            step_map[gpu_id] = {}
            self.reset_node_id()
            for tb in gpu.findall('tb'):
                tb_id = int(tb.attrib['id'])
                node_map[gpu_id][tb_id] = {}
                step_map[gpu_id][tb_id] = {}
                for step in tb.findall('step'):
                    step_id = int(step.attrib['s'])
                    step_map[gpu_id][tb_id][step_id] = step
                    if step.attrib['type'] == "s":
                        node = self.get_send_node(tb, step_id)
                        node_map[gpu_id][tb_id][step_id] = node
                    elif step.attrib['type'] == "r":
                        node = self.get_recv_node(tb, step_id)
                        node_map[gpu_id][tb_id][step_id] = node
                    elif step.attrib['type'] == "rrc":
                        node = self.get_recv_node(tb, step_id)
                        node_map[gpu_id][tb_id][step_id] = [node]
                        node = self.get_comp_node(tb, step_id)
                        node_map[gpu_id][tb_id][step_id].append(node)

        # For each ET Trace node, add the parent dependency information, then write to ET Trace file.
        for gpu_id in node_map:
            output_filename = "%s.%s.et" % (self.output_filename, gpu_id)
            with open(output_filename, "wb") as g:
                global_metadata = self.create_global_metadata()
                encode_message(g, global_metadata)
                for tb_id in node_map[gpu_id]:
                    prev_node = Node()
                    for step_id, et_node in node_map[gpu_id][tb_id].items():
                        if type(et_node) is list:
                            # RRC
                            recv_node = et_node[0]
                            comp_node = et_node[1]
                            # Parent by control
                            if step_id != 0:
                                self.add_parent(recv_node, prev_node)
                            self.add_parent(comp_node, recv_node)

                            # Parent by data dependency
                            step = step_map[gpu_id][tb_id][step_id]
                            dep_tb_id = int(step.attrib['depid'])
                            dep_step_id = int(step.attrib['deps'])

                            if dep_tb_id != -1:
                                dep_node = node_map[gpu_id][dep_tb_id][dep_step_id]
                                if type(dep_node) is list:
                                    self.add_parent(recv_node, dep_node[1])
                                else:
                                    self.add_parent(recv_node, dep_node)
                            encode_message(g, recv_node)
                            encode_message(g, comp_node)
                            if gpu_id == 0:
                                print('encode node', recv_node)
                                print('encode node', comp_node)
                            prev_node = comp_node
                            continue

                        # Parent by control
                        if step_id != 0:
                            self.add_parent(et_node, prev_node)
                        # Parent by data dependency
                        step = step_map[gpu_id][tb_id][step_id]
                        dep_tb_id = int(step.attrib['depid'])
                        dep_step_id = int(step.attrib['deps'])
                        if dep_tb_id != -1:
                            dep_node = node_map[gpu_id][dep_tb_id][dep_step_id]
                            if type(dep_node) is list:
                                # When msccl instr is 'rrc', dep_node is a list holding the corresponding COMM_RECV and COMP nodes. 
                                # We only add the COMP node as the parent (COMM_RECV is already added as the parent to COMP)
                                self.add_parent(et_node, dep_node[1])
                            else:
                                self.add_parent(et_node, dep_node)
                        encode_message(g, et_node)
                        if gpu_id == 0:
                            print('encode node', et_node)
                        prev_node = et_node

