import math
import pandas as pd

def generate_flit_trace_v2(tensor_trace_path, flit_trace_path, flit_size, injection_rate, radix):
    """
    Generates a flit trace file from a tensor trace file, where each flit is a separate packet.

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
            # Skip header lines
            next(f)
            next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    # The last 5 columns have a fixed format. The tensor name is everything in between.
                    timestamp = parts[0]
                    duration = parts[1]
                    core = parts[2]
                    origin = parts[3]
                    size = parts[-1]
                    tensor = " ".join(parts[4:-1])
                    
                    data.append({
                        'Timestamp': timestamp,
                        'Duration': duration,
                        'Core': core,
                        'Origin': origin,
                        'Tensor': tensor,
                        'Size': size
                    })
        df = pd.DataFrame(data)

    except FileNotFoundError:
        print(f"Error: The file {tensor_trace_path} was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading {tensor_trace_path}: {e}")
        return

    # Convert relevant columns to numeric types
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df['Origin'] = pd.to_numeric(df['Origin'], errors='coerce')
    df['Core'] = pd.to_numeric(df['Core'], errors='coerce')
    df['Size'] = pd.to_numeric(df['Size'], errors='coerce')
    df.dropna(subset=['Timestamp', 'Origin', 'Core', 'Size'], inplace=True)

    df['Origin'] = df['Origin'].astype(int)
    df['Core'] = df['Core'].astype(int)

    # Filter out rows where Origin and Core are the same
    df = df[df['Origin'] != df['Core']]
    
    flits = []
    packet_id = 0
    current_time = 0.0
    
    # Time to inject one flit across the network
    time_increment = 1.0 / (radix * injection_rate)

    # Sort by original timestamp to process in order
    df.sort_values(by='Timestamp', inplace=True)

    for _, row in df.iterrows():
        # Ensure the injection time is not earlier than the tensor's appearance time
        current_time = max(current_time, float(row['Timestamp']))
        
        source = row['Origin']
        destination = row['Core']
        tensor_size = row['Size']

        if tensor_size <= 0:
            continue

        num_flits = math.ceil(tensor_size / flit_size)
        
        for _ in range(num_flits):
            flits.append({
                'Time': int(round(current_time)),
                'Source': source,
                'Destination': destination,
                'PacketID': packet_id,
                'FlitPosition': "0/0",
                'IsHead': 1,
                'IsTail': 1
            })
            
            # Each flit is a new packet, so increment the packet_id
            packet_id += 1
            # Increment time for the next flit injection
            current_time += time_increment

    with open(flit_trace_path, 'w') as f:
        # Write header
        f.write(f"{'Time':<8}{'Source':<8}{'Destination':<12}{'PacketType':<12}{'FlitID':<8}{'PacketID':<12}{'FlitPosition':<16}{'IsHead':<8}{'IsTail':<8}\n")
        for flit in flits:
            # For v2, FlitID and PacketID are the same
            flit_id = flit['PacketID']
            f.write(
                f"{flit['Time']:<8}"
                f"{flit['Source']:<8}"
                f"{flit['Destination']:<12}"
                f"{'ANY':<12}"
                f"{flit_id:<8}"
                f"{flit['PacketID']:<12}"
                f"{flit['FlitPosition']:<16}"
                f"{flit['IsHead']:<8}"
                f"{flit['IsTail']:<8}\n"
            )
    
    print(f"Successfully generated {flit_trace_path} with {len(flits)} flits (v2 format).")

def main():
    """
    Main function to set parameters and run the flit trace generation.
    """
    flit_size = 128
    injection_rate = 0.01
    radix = 64

    tensor_trace_file = './tensor_trace_v2.txt'
    flit_trace_file = './flit_trace_v2.txt'

    generate_flit_trace_v2(tensor_trace_file, flit_trace_file, flit_size, injection_rate, radix)

if __name__ == "__main__":
    main()
