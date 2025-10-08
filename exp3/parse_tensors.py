import json
import re

def parse_tensor_trace(file_path):
    """
    Parses the scme.json file to trace tensor usage, including origin.

    Args:
        file_path (str): The path to the scme.json file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a tensor usage event.
    """
    try:
        with open(file_path, 'r') as f:
            file_content = f.read()
            # Fix for invalid json format. It may have trailing commas or empty objects.
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

    tensor_events = []
    wi_origin_map = {}
    
    # Sort data by timestamp to process events in order
    sorted_data = sorted([evt for evt in data if evt and 'ts' in evt], key=lambda x: x['ts'])

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
                    
                    origin = None
                    if tensor_type in ['W', 'I']:
                        if tensor_name not in wi_origin_map:
                            wi_origin_map[tensor_name] = tid
                        origin = wi_origin_map[tensor_name]
                    elif tensor_type == 'O':
                        origin = o_origin_map.get(tensor_name, 'N/A')

                    tensor_events.append({
                        "tensor": tensor_name,
                        "size": tensor_size,
                        "core": tid,
                        "ts": ts,
                        "dur": dur,
                        "origin": origin
                    })

    return tensor_events

def main():
    """
    Main function to parse the tensor trace and save it to a file.
    """
    file_path = './outputs/SIMD_array64-4_convs-lbl-genetic_algorithm/scme.json'
    output_path = './tensor_trace.txt'
    tensor_usage = parse_tensor_trace(file_path)

    if not tensor_usage:
        print("No tensor usage data found or failed to parse the file.")
        return

    with open(output_path, 'w') as f:
        f.write(f"{'Timestamp':<12} {'Duration':<10} {'Core':<6} {'Origin':<6} {'Tensor':<20} {'Size':<10}\n")
        f.write("="*70 + "\n")
        for event in tensor_usage:
            f.write(f"{event['ts']:<12} {event['dur']:<10} {event['core']:<6} {str(event['origin']):<6} {event['tensor']:<20} {event['size']:<10}\n")
    
    print(f"Tensor trace saved to {output_path}")

if __name__ == "__main__":
    main()
