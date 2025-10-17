import json
import re
import random
from collections import defaultdict

def parse_tensor_trace_v2(file_path):
    """
    Parses the scme.json file to trace tensor usage, including a randomly assigned origin
    for 'W' and 'I' tensors to distribute the load.

    Args:
        file_path (str): The path to the scme.json file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a tensor usage event.
    """
    try:
        with open(file_path, 'r') as f:
            file_content = f.read()
            # Fix for invalid json format
            file_content = re.sub(r',\s*([\}\]])', r'\1', file_content)
            file_content = re.sub(r'{\s*},?', '', file_content)
            if not file_content.strip().startswith('['):
                file_content = '[' + file_content
            if not file_content.strip().endswith(']'):
                last_brace_index = file_content.rfind('}')
                if last_brace_index != -1:
                    file_content = file_content[:last_brace_index+1] + ']'
            data = json.loads(file_content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return []

    # Pre-process to get origins for 'O' tensors from 'block' events
    o_origin_map = {}
    for event in data:
        if event and event.get("cat") == "block":
            args = event.get("args", {})
            sender = args.get("Sender")
            tensors = args.get("Tensors", {})
            if sender and tensors:
                sender_core = re.search(r'Core\((\d+)\)', sender)
                if sender_core:
                    for tensor_name in tensors.keys():
                        o_origin_map[tensor_name] = int(sender_core.group(1))

    # Sort data by timestamp to process events in chronological order
    sorted_data = sorted([evt for evt in data if evt and 'ts' in evt], key=lambda x: x['ts'])

    # --- New Logic: Multi-pass approach for W/I tensors ---

    # 1. First Pass: Collect all events and potential origins for W/I tensors
    all_events = []
    potential_wi_origins = defaultdict(set)

    for event in sorted_data:
        if event.get("cat") == "compute":
            ts = event.get("ts")
            dur = event.get("dur")
            tid = event.get("tid")
            tensors = event.get("args", {}).get("Tensors", {})
            if tensors:
                for tensor_name, tensor_size in tensors.items():
                    tensor_type_match = re.search(r',\s*([WIO])\)', tensor_name)
                    tensor_type = tensor_type_match.group(1) if tensor_type_match else None
                    
                    # Store the raw event
                    all_events.append({
                        "tensor": tensor_name,
                        "size": tensor_size,
                        "core": tid,
                        "ts": ts,
                        "dur": dur,
                        "type": tensor_type,
                        "origin": None # To be filled later
                    })

                    # Collect potential origins
                    if tensor_type in ['W', 'I']:
                        potential_wi_origins[tensor_name].add(tid)

    # 2. Random Selection: Choose a single origin for each W/I tensor
    final_wi_origins = {}
    for tensor_name, origins in potential_wi_origins.items():
        if origins:
            final_wi_origins[tensor_name] = random.choice(list(origins))

    # 3. Final Assignment: Update all events with the chosen origins
    for event in all_events:
        if event['type'] in ['W', 'I']:
            event['origin'] = final_wi_origins.get(event['tensor'])
        elif event['type'] == 'O':
            event['origin'] = o_origin_map.get(event['tensor'], 'N/A')

    return all_events

def main():
    """
    Main function to parse the tensor trace and save it to a file.
    """
    file_path = './outputs/SIMD_array64-my_custom_model_simplified-lbl-genetic_algorithm/scme.json'
    output_path = './tensor_trace_v2.txt'
    tensor_usage = parse_tensor_trace_v2(file_path)

    if not tensor_usage:
        print("No tensor usage data found or failed to parse the file.")
        return

    # Sort final results by timestamp for coherent output
    tensor_usage.sort(key=lambda x: x['ts'])

    with open(output_path, 'w') as f:
        f.write(f"{'Timestamp':<12} {'Duration':<10} {'Core':<6} {'Origin':<6} {'Tensor':<20} {'Size':<10}\n")
        f.write("="*70 + "\n")
        for event in tensor_usage:
            f.write(f"{event['ts']:<12} {event['dur']:<10} {event['core']:<6} {str(event['origin']):<6} {event['tensor']:<20} {event['size']:<10}\n")
    
    print(f"Tensor trace (v2) saved to {output_path}")

if __name__ == "__main__":
    main()
