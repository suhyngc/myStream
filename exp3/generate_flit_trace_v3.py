import math
import pandas as pd
from collections import defaultdict

def generate_flit_trace_v3(tensor_trace_path, flit_trace_path, flit_size, injection_rate, radix):
    """
    Generates a flit trace file from a tensor trace file, where flits from the same
    origin are sequential, but flits from different origins at the same timestamp can be parallel.
    Each flit is treated as a separate packet.

    Args:
        tensor_trace_path (str): Path to the tensor_trace.txt file.
        flit_trace_path (str): Path to save the output flit_trace.txt file.
        flit_size (int): The size of a single flit.
        injection_rate (float): The average number of flits injected per cycle per port.
        radix (int): The number of injection ports in the system.
    """
    try:
        # Manually read and parse the file to build a list of dictionaries
        data = []
        with open(tensor_trace_path, 'r') as f:
            next(f) # Skip header
            next(f) # Skip separator
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    timestamp = parts[0]
                    core = parts[2]
                    origin = parts[3]
                    size = parts[-1]
                    tensor = " ".join(parts[4:-1])
                    data.append({'Timestamp': timestamp, 'Core': core, 'Origin': origin, 'Size': size})
        df = pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: The file {tensor_trace_path} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading {tensor_trace_path}: {e}")
        return

    # Convert columns to numeric types
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df['Origin'] = pd.to_numeric(df['Origin'], errors='coerce')
    df['Core'] = pd.to_numeric(df['Core'], errors='coerce')
    df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
    df.dropna(subset=['Timestamp', 'Origin', 'Core', 'Size'], inplace=True)

    df['Origin'] = df['Origin'].astype(int)
    df['Core'] = df['Core'].astype(int)

    # Filter out rows where Origin and Core are the same
    df = df[df['Origin'] != df['Core']]
    
    # Sort by timestamp to process tensors in chronological order
    df.sort_values(by='Timestamp', inplace=True)

    flits = []
    packet_id = 0
    
    # Dictionary to track when an origin core is done generating its flits
    origin_finish_time = defaultdict(float)
    # Single variable to track the next available time for network-wide injection
    global_next_injection_time = 0.0
    
    # Time to inject one flit across the entire network
    time_increment = 1.0 / (radix * injection_rate)

    for _, row in df.iterrows():
        source = row['Origin']
        destination = row['Core']
        tensor_size = row['Size']

        if tensor_size <= 0:
            continue

        # Determine when the core is ready to start generating this tensor's flits.
        # The absolute timestamp of the tensor is ignored.
        core_ready_time = origin_finish_time[source]

        num_flits = math.ceil(tensor_size / flit_size)
        
        for _ in range(num_flits):
            # The actual injection time is the later of when the core is ready 
            # AND the global network is ready.
            injection_time = max(core_ready_time, global_next_injection_time)

            flits.append({
                'Time': int(round(injection_time)),
                'Source': source,
                'Destination': destination,
                'PacketID': packet_id,
            })
            
            packet_id += 1
            # The next global injection can only happen after this one
            global_next_injection_time = injection_time + time_increment
        
        # Update the finish time for this origin core. It's now ready for a new tensor.
        # We use the injection time of the last flit of the current tensor.
        origin_finish_time[source] = global_next_injection_time

    # Sort all generated flits by their injection time for the final output file
    flits.sort(key=lambda x: x['Time'])

    with open(flit_trace_path, 'w') as f:
        f.write(f"{'Time':<8}{'Source':<8}{'Destination':<12}{'PacketType':<12}{'FlitID':<8}{'PacketID':<12}{'FlitPosition':<16}{'IsHead':<8}{'IsTail':<8}\n")
        for flit in flits:
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
    
    print(f"Successfully generated {flit_trace_path} with {len(flits)} flits (v3 format).")

def main():
    """
    Main function to set parameters and run the flit trace generation.
    """
    flit_size = 128
    injection_rate = 0.01
    radix = 64

    # Using the v2 tensor trace with randomized origins
    tensor_trace_file = './tensor_trace_v3.txt'
    flit_trace_file = './flit_trace_v3.txt'

    generate_flit_trace_v3(tensor_trace_file, flit_trace_file, flit_size, injection_rate, radix)

if __name__ == "__main__":
    main()
