import math
import pandas as pd
from collections import defaultdict, deque

def generate_flit_trace_v4(tensor_trace_path, flit_trace_path, flit_size, injection_rate, radix):
    """
    Generates a flit trace file using a more realistic parallel injection model.
    Flits from different sources can be injected in parallel, respecting the global
    injection rate, while flits from the same source remain sequential.

    Args:
        tensor_trace_path (str): Path to the tensor_trace.txt file.
        flit_trace_path (str): Path to save the output flit_trace.txt file.
        flit_size (int): The size of a single flit.
        injection_rate (float): The average number of flits injected per cycle per port.
        radix (int): The number of injection ports in the system.
    """
    try:
        # Manually read and parse the file
        data = []
        with open(tensor_trace_path, 'r') as f:
            next(f); next(f) # Skip headers
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    data.append({
                        'Timestamp': parts[0], 'Core': parts[2], 'Origin': parts[3],
                        'Size': parts[-1], 'Tensor': " ".join(parts[4:-1])
                    })
        df = pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: The file {tensor_trace_path} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading {tensor_trace_path}: {e}")
        return

    # --- Data Preparation ---
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df['Origin'] = pd.to_numeric(df['Origin'], errors='coerce')
    df['Core'] = pd.to_numeric(df['Core'], errors='coerce')
    df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
    df.dropna(subset=['Timestamp', 'Origin', 'Core', 'Size'], inplace=True)
    df['Origin'] = df['Origin'].astype(int)
    df['Core'] = df['Core'].astype(int)
    df = df[df['Origin'] != df['Core']]
    df.sort_values(by=['Timestamp', 'Origin'], inplace=True)

    # --- V4 Logic: Event-driven simulation with source queues ---

    # 1. Create source queues for all flits
    source_queues = defaultdict(deque)
    packet_id_counter = 0
    for _, row in df.iterrows():
        num_flits = math.ceil(row['Size'] / flit_size)
        if num_flits > 0:
            for _ in range(num_flits):
                source_queues[row['Origin']].append({
                    'Source': row['Origin'],
                    'Destination': row['Core'],
                    'PacketID': packet_id_counter
                })
                packet_id_counter += 1

    # 2. Simulate injection using a round-robin scheduler
    final_flit_trace = []
    global_next_injection_time = 0.0
    time_increment = 1.0 / (radix * injection_rate)
    
    active_sources = list(source_queues.keys())
    
    while any(source_queues.values()):
        for source_id in active_sources:
            if source_queues[source_id]:
                flit = source_queues[source_id].popleft()
                
                # Assign injection time and add to final trace
                flit['Time'] = int(round(global_next_injection_time))
                final_flit_trace.append(flit)
                
                # Update the global network time
                global_next_injection_time += time_increment

    # Sort by time as a final guarantee of chronological order
    final_flit_trace.sort(key=lambda x: x['Time'])

    # 3. Write to output file
    with open(flit_trace_path, 'w') as f:
        f.write(f"{'Time':<8}{'Source':<8}{'Destination':<12}{'PacketType':<12}{'FlitID':<8}{'PacketID':<12}{'FlitPosition':<16}{'IsHead':<8}{'IsTail':<8}\n")
        for flit in final_flit_trace:
            flit_id = flit['PacketID']
            f.write(
                f"{flit['Time']:<8}"
                f"{flit['Source']:<8}"
                f"{flit['Destination']:<12}"
                f"{'ANY':<12}"
                f"{flit_id:<8}"
                f"{flit['PacketID']:<12}"
                f"{'0/0':<16}"
                f"{1:<8}"
                f"{1:<8}\n"
            )
    
    print(f"Successfully generated {flit_trace_path} with {len(final_flit_trace)} flits (v4 format).")

def main():
    flit_size = 128
    injection_rate = 0.01
    radix = 64

    tensor_trace_file = './tensor_trace_v4.txt'
    flit_trace_file = './flit_trace_v4.txt'

    generate_flit_trace_v4(tensor_trace_file, flit_trace_file, flit_size, injection_rate, radix)

if __name__ == "__main__":
    main()
