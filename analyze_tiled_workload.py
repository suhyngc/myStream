#!/usr/bin/env python3
import pickle
from pprint import pprint
import os
from stream.workload.onnx_workload import ComputationNodeWorkload

def pickle_load(file_path):
    """Load a pickle file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def analyze_tiled_workload(pickle_path):
    """Analyze the tiled workload pickle file and print its attributes"""
    print(f"Analyzing tiled workload from: {pickle_path}")
    
    # Load the pickle file
    tiled_workload = pickle_load(pickle_path)
    if not isinstance(tiled_workload, ComputationNodeWorkload):
        print("Error: Loaded file is not a ComputationNodeWorkload")
        return
    
    print("\n=== Basic Workload Information ===")
    print(f"Number of nodes: {len(tiled_workload.node_list)}")
    print(f"Number of edges: {len(tiled_workload.edges)}")
    
    # Analyze nodes
    print("\n=== Node Analysis ===")
    for node in tiled_workload.node_list:
        print(f"\nNode ID: {node.id}, Sub ID: {node.sub_id}")
        print(f"Node Type: {node.type}")
        print(f"Node Name: {node.name}")
        print(f"Group ID: {node.group}")
        print(f"Layer Dimensions: {node.layer_dims}")
        print(f"Layer Dimension Sizes: {node.layer_dim_sizes}")
        print(f"Intra-core Tiling: {node.intra_core_tiling}")
        print(f"Inter-core Tiling: {node.inter_core_tiling}")
        print(f"Loop Ranges: {node.loop_ranges}")
        
        # Compute/Memory timing information
        if hasattr(node, 'compute_cycles'):
            print(f"Compute Cycles: {node.compute_cycles}")
        if hasattr(node, 'memory_cycles'):
            print(f"Memory Access Cycles: {node.memory_cycles}")
        if hasattr(node, 'latency'):
            print(f"Total Latency: {node.latency}")
            
        print(f"Operand Tensors:")
        for op, tensor in node.operand_tensors.items():
            print(f"  - {op}: {tensor}")
            # Check for operand-specific timing
            if hasattr(tensor, 'read_cycles'):
                print(f"    Read Cycles: {tensor.read_cycles}")
            if hasattr(tensor, 'write_cycles'):
                print(f"    Write Cycles: {tensor.write_cycles}")
                
        print(f"Produces Final Output: {node.produces_final_output}")
        print(f"Data Produced Unique: {node.data_produced_unique}")
        
        if hasattr(node, 'chosen_core_allocation'):
            print(f"Chosen Core Allocation: {node.chosen_core_allocation}")
        
        print(f"Input Operands: {node.input_operands}")
        print(f"Output Operand: {node.output_operand}")
        print(f"Constant Operands: {node.constant_operands}")
        print("-" * 50)

    # Analyze edges
    print("\n=== Edge Analysis ===")
    for edge in tiled_workload.edges:
        source = edge[0]
        target = edge[1]
        attrs = edge[2] if len(edge) > 2 else {}
        print(f"\nEdge: {source.id}:{source.sub_id} -> {target.id}:{target.sub_id}")
        
        # Communication timing analysis
        if attrs:
            print(f"Edge Attributes: {attrs}")
            # Check for communication timing information
            if 'communication_cycles' in attrs:
                print(f"Communication Cycles: {attrs['communication_cycles']}")
            if 'bandwidth' in attrs:
                print(f"Bandwidth: {attrs['bandwidth']}")
            if 'latency' in attrs:
                print(f"Communication Latency: {attrs['latency']}")
            
            # Check for data transfer information
            if 'data_volume' in attrs:
                print(f"Data Transfer Volume: {attrs['data_volume']}")
            if 'transfer_cycles' in attrs:
                print(f"Data Transfer Cycles: {attrs['transfer_cycles']}")

def main():
    pickle_path = "/pool0/suhyeong/06_3D_interconnection/stream/lab1_copy/outputs/hda_bus-4_convs-lbl-genetic_algorithm/tiled_workload.pickle"
    
    if not os.path.exists(pickle_path):
        print(f"Error: File not found at {pickle_path}")
        return
        
    analyze_tiled_workload(pickle_path)

if __name__ == "__main__":
    main()
