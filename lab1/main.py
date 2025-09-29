import logging as _logging
import os
import re
import sys

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
            # Get the split information if this tensor was split
            if hasattr(node, 'intra_core_tiling') and node.intra_core_tiling:
                split_dims = [dim for dim, _ in node.intra_core_tiling if dim in tensor.loop_dimensions]
                ranges = {dim: node.loop_ranges[dim] for dim in split_dims}
                original_size = tensor.size
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

# Function call
usage_info = analyze_tensor_partitions_and_usage(scme)

# Using the results
print("\nDetailed Tensor Usage Analysis:")
print("================================")

# Process and print results
from collections import defaultdict
tensor_groups = defaultdict(list)
for info in usage_info:
    tensor_groups[info['tensor_hash']].append(info)

for tensor_hash, operations in tensor_groups.items():
    print(f"\nTensor Hash: {tensor_hash}")
    print("------------------------")
    for op in sorted(operations, key=lambda x: x['time_start']):
        if op.get('operation') == 'transfer':
            print(f"Time [{op['time_start']}-{op['time_end']}]: Transfer")
            if 'from_core' in op and 'to_core' in op:
                print(f"  From Core {op['from_core']} to Core {op['to_core']}")
            print(f"  Size: {op['partition_size']}/{op['total_size']} bits")
            if 'bandwidth' in op:
                print(f"  Bandwidth: {op['bandwidth']} bits/cycle")
            if 'energy' in op:
                print(f"  Energy: {op['energy']:.2e}")
        else:
            print(f"Time [{op['time_start']}-{op['time_end']}]: {op['operation'].capitalize()}")
            print(f"  Core: {op['core_id']}")
            print(f"  Node: {op['node_id']}")
            print(f"  Size: {op['partition_size']}/{op['total_size']} bits")
            if op['split_ranges']:
                print(f"  Split ranges: {op['split_ranges']}")
        print()
