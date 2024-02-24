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

HARDCODE_COMM_SIZE = int(1024 * 1024 / 4)
HARDCODE_LOCAL_BW = 50
# 1000 b/c microsecond to nanosecond. Refer to Workload::issue_replay
HARDCODE_COMP_TIME_NS = int (3 * int(HARDCODE_COMM_SIZE / HARDCODE_LOCAL_BW) / 1000)

class Layer:
    def __init__(
        self,
        line: str
    ) -> None:
        try:
            col = line.strip().split()
            self.name = col[0]

            # forward
            self.fwd_comp_time = int(col[2])
            self.fwd_comm_type = str(col[3])
            self.fwd_comm_size = int(col[4])
            self.fwd_comp_node = None
            self.fwd_comm_node = None

            # backward input gradient
            self.bwd_ig_comp_time = int(col[5])
            self.bwd_ig_comm_type = str(col[6])
            self.bwd_ig_comm_size = int(col[7])
            self.bwd_ig_comp_node = None
            self.bwd_ig_comm_node = None

            # backward weight gradient
            self.bwd_wg_comp_time = int(col[8])
            self.bwd_wg_comm_type = str(col[9])
            self.bwd_wg_comm_size = int(col[10])
            self.bwd_wg_update_time = str(col[11])
            self.bwd_wg_comp_node = None
            self.bwd_wg_comm_node = None
        except Exception:
            raise ValueError(f"Cannot parse the following layer -- \"{line}\"")

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

    def get_global_metadata(self):
        input_text = ""
        with open(self.input_filename, "r") as input_file:
            input_text = input_file.read()
        attr = [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="input_file", string_val=input_text)
        ]
        metadata = GlobalMetadata(attr=attr)
        return metadata


    def get_node(
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

    def reset_node_id(
            self
    ): self.next_node_id = 0

    def get_comp_node(
        self,
        gpu_id: int,
        tb_xml_node: ElementTree.Element
    ) -> Any:
        tb_id = tb_xml_node.attrib['id']
        node = self.get_node(f"COMP_NODE_gpu{gpu_id}_tb{tb_id}",
                             COMP_NODE)
        node.duration_micros = HARDCODE_COMP_TIME_NS
        return node

    def get_send_node (
        self, 
        gpu_id: int,
        tb_xml_node: ElementTree.Element
    ) -> Any:
        tb_id = tb_xml_node.attrib['id']
        layer_name = f'COMM_SEND_NODE_gpu{gpu_id}_tb{tb_id}'
        src = gpu_id
        dst = int(tb_xml_node.attrib['send'])
        tag = int(tb_xml_node.attrib['chan'])
        size = HARDCODE_COMM_SIZE

        node = self.get_node(layer_name, COMM_SEND_NODE)
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
        gpu_id: int,
        tb_xml_node: ElementTree.Element
    ) -> Any:
        tb_id = tb_xml_node.attrib['id']
        layer_name = f'COMM_RECV_NODE_gpu{gpu_id}_tb{tb_id}'
        src = int(tb_xml_node.attrib['recv'])
        dst = gpu_id
        tag = int(tb_xml_node.attrib['chan'])
        size = HARDCODE_COMM_SIZE

        node = self.get_node(layer_name, COMM_RECV_NODE)
        node.attr.append(ChakraAttr(name="comm_type",
                                    int64_val=COMM_RECV_NODE))
        node.attr.append(ChakraAttr(name="comm_size",
                                    uint64_val=size))
        node.attr.append(ChakraAttr(name="comm_src",
                                    uint64_val=src))
        node.attr.append(ChakraAttr(name="comm_tag",
                                    uint64_val=tag))
        return node

    def add_parent(
        self,
        child_node: Any,
        parent_node: Any
    ) -> None:
        child_node.data_deps.append(parent_node.id)
# diff ctrl dep, data dep
        
    def convert(self) -> None:
        tree = ElementTree.parse('./et_converter/allreduce_ring_4_old_format.xml')
        root = tree.getroot()
        for gpu in root.findall('gpu'):
            gpu_id = gpu.attrib['id']
            for tb in gpu.findall('tb'):
                tb_id = tb.attrib['id']
                output_filename = "%s.%s.et" % (self.output_filename, gpu_id)
                with open(output_filename, "wb") as g:
                    self.reset_node_id()
                    global_metadata = self.get_global_metadata()
                    encode_message(g, global_metadata)
                    prev_node = Node()
                    for step in tb.findall('step'):
                        if step.attrib['type'] == "s":
                            print('s')
                            node = self.get_send_node(gpu_id, tb)
                        elif step.attrib['type'] == "r":
                            print('r')
                            node = self.get_recv_node(gpu_id, tb)
                        elif step.attrib['type'] == "rrc":
                            print('rrc')
                            node = self.get_recv_node(gpu_id, tb)
                            if node.id != 0:
                                self.add_parent(node, prev_node)
                                print(f'add {prev_node.id} as parent to {node.id}')
                            prev_node = node 
                            encode_message(g, node)
                            node = self.get_comp_node(gpu_id, tb)
                        if node.id != 0:
                            self.add_parent(node, prev_node)
                            print(f'add {prev_node.id} as parent to {node.id}')
                        prev_node = node
                        encode_message(g, node)
        
