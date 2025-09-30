import logging as _logging
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.getcwd())  # Insert main folder in path
from zigzag.utils import pickle_load

from stream.api import optimize_allocation_ga
from stream.utils import CostModelEvaluationLUT
from stream.visualization.memory_usage import plot_memory_usage
from stream.visualization.perfetto import convert_scme_to_perfetto_json

############################## Initialize the logger ##############################
_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)
logger = _logging.getLogger(__name__)
####################################################################################

############################## Provide inputs ######################################
workload_path = "lab1/inputs/workload/4_convs.onnx"
accelerator = "lab1/inputs/hardware/hda_bus.yaml"
mapping_path = "lab1/inputs/mapping/mapping.yaml"
mode = "lbl"
nb_ga_generations = 4
nb_ga_individuals = 4
####################################################################################

################################## Parsing #########################################
hw_name = accelerator.split("/")[-1].split(".")[0]
wl_name = re.split(r"/|\.", workload_path)[-1]
if wl_name == "onnx":
    wl_name = re.split(r"/|\.", workload_path)[-2]
experiment_id = f"{hw_name}-{wl_name}-{mode}-genetic_algorithm"
####################################################################################

##############PLOTTING###############
plot_file_name = f"-{experiment_id}-"
plot_full_schedule = True
draw_dependencies = True
plot_data_transfer = True
section_start_percent = (0,)
percent_shown = (100,)
#####################################


################################PATHS################################
output_folder = f"lab1/outputs/{experiment_id}"
timeline_fig_path_plotly = f"{output_folder}/schedule.html"
memory_fig_path = f"{output_folder}/memory.png"
json_path = f"{output_folder}/scme.json"
scme_path = f"{output_folder}/scme.pickle"
#####################################################################

if not os.path.exists(scme_path):
    scme = optimize_allocation_ga(
        hardware=accelerator,
        workload=workload_path,
        mapping=mapping_path,
        mode=mode,
        layer_stacks=None,
        nb_ga_generations=nb_ga_generations,
        nb_ga_individuals=nb_ga_individuals,
        experiment_id=experiment_id,
        output_path="lab1/outputs",
        skip_if_exists=True,
    )
else:
    scme = pickle_load(scme_path)

# Load in the CostModelEvaluationLUT from the run
cost_lut_path = f"{output_folder}/cost_lut.pickle"
cost_lut = CostModelEvaluationLUT(cost_lut_path)

# Plotting memory usage of best SCME
plot_memory_usage(scme, section_start_percent, percent_shown, fig_path=memory_fig_path, show_dram=True)

# Save json for perfetto visualization (Visualize at http://ui.perfetto.dev/)
convert_scme_to_perfetto_json(scme, cost_lut, json_path=json_path)

# Function definition
def analyze_tensor_partitions_and_usage(scme):
    usage_info = []
    
    # Get nodes from workload
    G = scme.workload  # Get the workload graph
    
    # Track all tensor operations including splits and transfers
    for node in G.node_list:
        core_id = node.chosen_core_allocation
        start_time = node.start
        end_time = node.end
        
        # Check each operand tensor used by this node
        for op, tensor in node.operand_tensors.items():
            # Get tensor shape and dimension information
            tensor_shape = {}
            if hasattr(tensor, 'shape'):
                tensor_shape = tensor.shape
            elif hasattr(tensor, 'loop_ranges_per_dim'):
                tensor_shape = {dim: (end-start) for dim, (start, end) in tensor.loop_ranges_per_dim.items()}
            
            # Get the tiling and loop information
            tiling_info = {}
            loop_ranges = {}
            loop_order = []
            
            if hasattr(node, 'intra_core_tiling') and node.intra_core_tiling:
                split_dims = [dim for dim, _ in node.intra_core_tiling if dim in tensor.loop_dimensions]
                ranges = {dim: node.loop_ranges[dim] for dim in split_dims}
                original_size = tensor.size
                
                # Collect tiling information
                for dim, (size, level) in node.intra_core_tiling:
                    if dim in tensor.loop_dimensions:
                        tiling_info[dim] = size
                
                # Get loop ranges and order
                if hasattr(node, 'loop_ranges'):
                    loop_ranges = node.loop_ranges
                if hasattr(node, 'loop_order'):
                    loop_order = node.loop_order
                
                # Calculate partition size based on split ratio
                partition_size = original_size
                for dim in split_dims:
                    full_range = tensor.loop_ranges_per_dim[dim]
                    used_range = node.loop_ranges[dim]
                    ratio = (used_range[1] - used_range[0]) / (full_range[1] - full_range[0])
                    partition_size = int(partition_size * ratio)
            else:
                ranges = {}
                partition_size = tensor.size
                
            usage_info.append({
                'tensor_id': tensor.id,
                'tensor_hash': tensor.equality_hash,
                'operation': 'read' if op != node.output_operand else 'write',
                'core_id': core_id,
                'time_start': start_time,
                'time_end': end_time,
                'total_size': tensor.size,
                'partition_size': partition_size,
                'split_ranges': ranges,
                'shape': tensor_shape,
                'tiling': tiling_info,
                'loop_ranges': loop_ranges,
                'loop_order': loop_order,
                'node_id': node.id
            })
    
    # Also track tensor transfers between cores
    for event in scme.accelerator.communication_manager.events:
        # Each communication event has multiple tasks (one per link in the path)
        for task in event.tasks:
            if hasattr(task, 'tensor'):  # CommunicationLinkEvent has single tensor
                usage_info.append({
                    'tensor_id': task.tensor.id,
                    'tensor_hash': task.tensor.equality_hash,
                    'operation': 'transfer',
                    'time_start': event.start,
                    'time_end': event.end,
                    'total_size': task.tensor.size,
                    'partition_size': task.tensor.size,  # For transfers, use full size
                    'energy': event.energy / len(event.tasks),  # Divide total energy by number of links
                    'type': event.type,
                    'activity': task.activity  # Percentage of link bandwidth used
                })
    
    return usage_info

# Create output directory for analysis if it doesn't exist
analysis_dir = f"{output_folder}/analysis"
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)

node_summary_file = f"{analysis_dir}/node_summary.txt"
tensor_analysis_file = f"{analysis_dir}/tensor_analysis.txt"

# First analyze workload structure
node_types = defaultdict(int)
node_details = defaultdict(list)

for node in scme.workload.node_list:
    node_type = type(node).__name__
    node_id = node.id
    node_types[node_type] += 1
    
    # Collect details about the node
    details = {
        'id': node_id,
        'core': node.chosen_core_allocation,
        'timing': (node.start, node.end),
        'input_sizes': {},
        'output_sizes': {},
    }
    
    # Get input tensor sizes
    for op, tensor in node.operand_tensors.items():
        if op == node.output_operand:
            details['output_sizes'][op] = tensor.size
        else:
            details['input_sizes'][op] = tensor.size
            
    node_details[node_type].append(details)

# Save node summary to file
with open(node_summary_file, 'w') as f:
    f.write("Node Types Summary:\n")
    f.write("=================\n")
    for node_type, count in node_types.items():
        f.write(f"{node_type}: {count} instances\n")

    f.write("\nDetailed Node Information:\n")
    f.write("========================\n")
    for node_type, details_list in node_details.items():
        f.write(f"\n{node_type} Nodes:\n")
        f.write("=" * (len(node_type) + 7) + "\n")
        
        for details in details_list:
            f.write(f"\nNode ID: {details['id']}\n")
            f.write(f"Core: {details['core']}\n")
            f.write(f"Execution: [{details['timing'][0]} - {details['timing'][1]}]\n")
            
            if details['input_sizes']:
                f.write("Input Tensors:\n")
                for op, size in details['input_sizes'].items():
                    f.write(f"  {op}: {size} bits\n")
            
            if details['output_sizes']:
                f.write("Output Tensors:\n")
                for op, size in details['output_sizes'].items():
                    f.write(f"  {op}: {size} bits\n")
            f.write("-" * 40 + "\n")

# Then run the original tensor analysis
usage_info = analyze_tensor_partitions_and_usage(scme)

print(f"\nAnalysis files have been saved to:")
print(f"1. Node Summary: {node_summary_file}")
print(f"2. Tensor Analysis: {tensor_analysis_file}")

# Process and save tensor analysis results
tensor_groups = defaultdict(list)
for info in usage_info:
    tensor_groups[info['tensor_hash']].append(info)

with open(tensor_analysis_file, 'w') as f:
    f.write("Tensor Usage Analysis:\n")
    f.write("=====================\n")
    
    for tensor_hash, operations in tensor_groups.items():
        f.write(f"\nTensor Hash: {tensor_hash}\n")
        f.write("------------------------\n")
        
        for op in sorted(operations, key=lambda x: x['time_start']):
            # Find the corresponding node information
            node_info = None
            if op.get('operation') != 'transfer':
                for node in scme.workload.node_list:
                    if node.id == op['node_id']:
                        node_info = node
                        break
            
            if op.get('operation') == 'transfer':
                f.write(f"Time [{op['time_start']}-{op['time_end']}]: Transfer\n")
                if 'from_core' in op and 'to_core' in op:
                    f.write(f"  From Core {op['from_core']} to Core {op['to_core']}\n")
                f.write(f"  Size: {op['partition_size']}/{op['total_size']} bits\n")
                if 'bandwidth' in op:
                    f.write(f"  Bandwidth: {op['bandwidth']} bits/cycle\n")
                if 'energy' in op:
                    f.write(f"  Energy: {op['energy']:.2e}\n")
            else:
                node_type = type(node_info).__name__ if node_info else "Unknown"
                f.write(f"Time [{op['time_start']}-{op['time_end']}]: {op['operation'].capitalize()}\n")
                f.write(f"  Core: {op['core_id']}\n")
                f.write(f"  Node: {op['node_id']} ({node_type})\n")
                if hasattr(node_info, 'equation'):
                    f.write(f"  Operation: {node_info.equation}\n")
                # Write tensor shape information
                if op['shape']:
                    f.write("  Tensor Shape:\n")
                    for dim, size in op['shape'].items():
                        f.write(f"    {dim}: {size}\n")
                
                # Write tiling information
                if op['tiling']:
                    f.write("  Tiling:\n")
                    for dim, size in op['tiling'].items():
                        f.write(f"    {dim}: {size}\n")
                
                # Write loop ranges and order
                if op['loop_ranges']:
                    f.write("  Loop Ranges:\n")
                    for dim, (start, end) in op['loop_ranges'].items():
                        f.write(f"    {dim}: [{start}:{end}]\n")
                
                if op['loop_order']:
                    f.write("  Loop Order: " + " -> ".join(op['loop_order']) + "\n")
                
                f.write(f"  Access Size: {op['partition_size']}/{op['total_size']} bits\n")
                if op['split_ranges']:
                    f.write("  Split Ranges:\n")
                    for dim, (start, end) in op['split_ranges'].items():
                        f.write(f"    {dim}: [{start}:{end}]\n")
                
                # Calculate and show memory footprint
                f.write("  Memory Access Pattern:\n")
                if op['tiling'] and op['loop_ranges']:
                    total_elements = 1
                    for dim, size in op['tiling'].items():
                        if dim in op['loop_ranges']:
                            start, end = op['loop_ranges'][dim]
                            iterations = (end - start) // size
                            total_elements *= iterations
                    f.write(f"    Number of tile accesses: {total_elements}\n")
                    f.write(f"    Bytes per tile: {op['partition_size']/8:.0f}\n")
                    f.write(f"    Total data movement: {(total_elements * op['partition_size'])/8:.0f} bytes\n")
                # Show which operand this tensor represents in the node's computation
                if node_info:
                    for op_name, tensor in node_info.operand_tensors.items():
                        if tensor.equality_hash == tensor_hash:
                            f.write(f"  Operand Role: {op_name}\n")
                            if op_name == node_info.output_operand:
                                f.write("  (This is the output tensor)\n")
                            break
            f.write("\n")

