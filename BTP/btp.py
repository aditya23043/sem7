import pickle
import re
import numpy as np
import subprocess
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import pandas as pd
import pulp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import minimize
import joblib
from sklearn.neural_network import MLPRegressor


# --- GLOBAL VARIABLES & PATHS ---

cnt = 000
margin = 0.40
slew = {"A":35,"B":35,"Cin":45}
cap = {"Sum":15,"Cout":25}
n_points = 1

vdd = 1.08
# --- CRITICAL PATH TARGET ---
# Extracted directly from Genus STA Timing Report
TARGET_PATH = {
    "start_pin": "A",
    "start_edge": "F",            # 'R' for Rise, 'F' for Fall
    "end_pin": "Sum",
    "end_edge": "F",              # 'R' for Rise, 'F' for Fall
    "side_inputs": {"B": 0.0,"Cin":0.0}     # The static stimuli needed to sensitize this path
}
Non_controlling_default = 1.0


netlist_filename = f"test.v"
input_dataset_filename = os.path.join("data", f"dataset{cnt}.scs")
spectre_netlist_filename = os.path.join("Spectre_SCS", f"netlist{cnt}.scs")
spectre_stimuli_filename = os.path.join("Spectre_SCS", f"stimuli{cnt}.scs")
spectre_meas_filename = os.path.join("Spectre_SCS", f"meas{cnt}.scs")
meas_read_file = 'top.mt0'
simulation_result= os.path.join("data", f"simulation{cnt}.csv")
config_filename = os.path.join("Spectre_SCS", f"config.scs")
Finalresults_filename = os.path.join("data",f"Final_run{cnt}.scs")

for i in slew:
    # Convert Genus 10-90% slew to Spectre 0-100% slew
    sta_slew_ps = slew.get(i, 0.0)
    slew[i] = sta_slew_ps / 0.8  # e.g., 80ps becomes 100ps
    # Convert to seconds for the SPICE paramset

def setup_environment():
    directories = ["data", "Spectre_SCS", "standard_cells"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

setup_environment()

class PMOS:
    def __init__(self, Name, W, L=0.45):
        self.Name = Name
        self.Type = "g45p1svt"
        self.W = W
        self.L = L

class NMOS:
    def __init__(self, Name, W, L=0.45):
        self.Name = Name
        self.Type = "g45n1svt"
        self.W = W
        self.L = L




def Verilogtraverser(filename):
    input_pins = []
    output_pins = []
    wires = []
    interconnections = {}

    with open(filename, "r") as file:
        lines = file.readlines()

    current_instance = None

    for line in lines:
        line = re.sub(r'//.*', '', line).strip()

        if not line or line.startswith("module") or line == "endmodule":
            continue

        if line.startswith("input"):
            clean_line = re.sub(r'input|\s*\[.*?\]\s*|;', '', line)
            for pin in clean_line.split(','):
                if pin.strip(): input_pins.append(pin.strip())

        elif line.startswith("output"):
            clean_line = re.sub(r'output|\s*\[.*?\]\s*|;', '', line)
            for pin in clean_line.split(','):
                if pin.strip(): output_pins.append(pin.strip())

        elif line.startswith("wire"):
            clean_line = re.sub(r'wire|\s*\[.*?\]\s*|;', '', line)
            for w in clean_line.split(','):
                if w.strip(): wires.append(w.strip())

        inst_match = re.search(r'([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_]+)\s*\(', line)
        if inst_match and not line.startswith("module"):
            gate_type = inst_match.group(1)
            instance_name = inst_match.group(2)
            current_instance = instance_name
            interconnections[current_instance] = [gate_type]

            # Initialize a buffer to hold the multi-line string
            instance_buffer = ""

        if current_instance:
            # Accumulate the current line into the buffer
            instance_buffer += " " + line

            # Only run the regex once we hit the end of the instance declaration
            if ");" in line:
                pin_matches = re.findall(r'\.\s*([a-zA-Z0-9_]+)\s*\(\s*([a-zA-Z0-9_\[\]]+)\s*\)', instance_buffer)

                for pin, net in pin_matches:
                    interconnections[current_instance].append([net, pin])

                current_instance = None

    return input_pins, output_pins, wires, interconnections


def Netlist_Dictionary_Generator(input_pins, output_pins, wires, interconnections):
    master_netlist = {}
    power_nets = ["VDD", "VSS", "VDD!", "VSS!"]

    for instance_name, data in interconnections.items():
        cellname = data[0]
        pin_mappings = data[1:]

        filepath = os.path.join("standard_cells", f"{cellname}.pkl")
        with open(filepath, "rb") as file:
            cell_dict = pickle.load(file)

        pin_lookup = {local_pin: global_net for global_net, local_pin in pin_mappings}

        for trans_name, trans_data in cell_dict.items():
            global_trans_name = f"{instance_name}_{trans_name}"

            master_netlist[global_trans_name] = {
                'type': trans_data['type'],
                'instance': trans_data['instance'],
                'node_name': trans_name
            }

            for terminal in ['D', 'G', 'S', 'B']:
                local_net = trans_data[terminal]

                if local_net in power_nets:
                    global_net = local_net.replace("!", "")
                elif local_net in pin_lookup:
                    global_net = pin_lookup[local_net]
                else:
                    global_net = f"{instance_name}_{local_net}"

                master_netlist[global_trans_name][terminal] = global_net

    return master_netlist

def generate_ml_dataset(master_netlist, input_pins, n_points, margin, filename):
    headers = []
    columns = []
    range_of_widhts = {}
    for global_trans_name, data in master_netlist.items():
        headers.append(global_trans_name)
        range_of_widhts[global_trans_name] = [0,0]
        # Multiply by 1e-6 to convert Microns to absolute Meters
        w_meters = float(data['instance'].W) * 1e-6

        # Check if any terminal of this transistor touches an input pin
        terminals = [data.get('D'), data.get('G'), data.get('S'), data.get('B')]
        touches_input = any(pin in input_pins for pin in terminals if pin is not None)

        if touches_input:
            # If it touches an input pin, fill the array with a CONSTANT width
            # including the +1 row for the baseline point
            column_data = np.full(n_points + 1, w_meters)
            range_of_widhts[global_trans_name] = [w_meters,w_meters]
            columns.append(column_data)
        else:
            # Randomize the widths for the n_points requested
            min_bound = w_meters * (1 - margin)
            max_bound = w_meters * (1 + margin)

            # Enforce physical PDK boundaries on the dictionary so PuLP obeys them
            clamped_min = max(min_bound, 120e-9)
            clamped_max = min(max_bound, 10e-6)
            
            range_of_widhts[global_trans_name] = [clamped_min, clamped_max]
            
            # Apply the same clip to the data matrix
            randomized_widths = np.random.uniform(min_bound, max_bound, n_points)
            clamped_widths = np.clip(randomized_widths, 120e-9, 10e-6)

            # Prepend the original w_meters to the very top of the randomized array
            column_data = np.concatenate(([w_meters], clamped_widths))
            columns.append(column_data)

    # Stack all columns vertically
    dataset_matrix = np.column_stack(columns)

    # Export to text/csv format
    np.savetxt(
        filename,
        dataset_matrix,
        delimiter=" ",
        header=" ".join(headers),
        comments="",
        fmt="%.6e"
    )
    return range_of_widhts

def generate_scs_netlist(master_netlist, input_pins, output_pins, filename):
    netlist_lines = []

    instances = list(master_netlist.keys())
    param_names = []
    param_names.extend(instances)

    for pin in input_pins:
        param_names.extend([f"slew_{pin}", f"start_{pin}", f"end_{pin}"])

    for pin in output_pins:
        param_names.append(f"cap_{pin}")

    param_line = "parameters " + " ".join([f"{p}=1u" for p in param_names])
    netlist_lines.append(param_line)
    netlist_lines.append("")

    stress_block = '''nf=ceil((({w}) - 5p) / (10u)) as=((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? ((((50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) * 120n) + ((({w}) / (ceil((({w}) - 5p) / (10u)))) * 50n)) + (floor(((ceil((({w}) - 5p) / (10u))) - 1) / 2.0) * (((((60n) - 0) + 60n) * 120n) + ((({w}) / (ceil((({w}) - 5p) / (10u)))) * 100n))) + ((((ceil((({w}) - 5p) / (10u))) / 2) - floor((ceil((({w}) - 5p) / (10u))) / 2) == 0) ? (((50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) * 120n) + ((({w}) / (ceil((({w}) - 5p) / (10u)))) * 50n)) : 0)) / 1 : (((100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n)) * (({w}) / (ceil((({w}) - 5p) / (10u))))) + (floor(((ceil((({w}) - 5p) / (10u))) - 1) / 2.0) * ((60n > (((60n) - 0) + 50n) ? 60n : (((60n) - 0) + 50n)) * (({w}) / (ceil((({w}) - 5p) / (10u)))))) + ((((ceil((({w}) - 5p) / (10u))) / 2) - floor((ceil((({w}) - 5p) / (10u))) / 2) == 0) ? ((100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n)) * (({w}) / (ceil((({w}) - 5p) / (10u))))) : 0)) / 1 \\
         ad=((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? ((floor((ceil((({w}) - 5p) / (10u))) / 2.0) * (((((60n) - 0) + 60n) * 120n) + ((({w}) / (ceil((({w}) - 5p) / (10u)))) * 100n))) + ((((ceil((({w}) - 5p) / (10u))) / 2) - floor((ceil((({w}) - 5p) / (10u))) / 2) != 0) ? (((50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) * 120n) + ((({w}) / (ceil((({w}) - 5p) / (10u)))) * 50n)) : 0)) / 1 : ((floor((ceil((({w}) - 5p) / (10u))) / 2.0) * ((60n > (((60n) - 0) + 50n) ? 60n : (((60n) - 0) + 50n)) * (({w}) / (ceil((({w}) - 5p) / (10u)))))) + ((((ceil((({w}) - 5p) / (10u))) / 2) - floor((ceil((({w}) - 5p) / (10u))) / 2) != 0) ? ((100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n)) * (({w}) / (ceil((({w}) - 5p) / (10u))))) : 0)) / 1 \\
         ps=((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (((2 * (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n))) + 340n) + (floor(((ceil((({w}) - 5p) / (10u))) - 1) / 2.0) * ((2 * (((60n) - 0) + 60n)) + 440n)) + ((((ceil((({w}) - 5p) / (10u))) / 2) - floor((ceil((({w}) - 5p) / (10u))) / 2) == 0) ? ((2 * (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n))) + 340n) : 0)) / 1 : (((2 * (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))) + (2 * (({w}) / (ceil((({w}) - 5p) / (10u)))))) + (floor(((ceil((({w}) - 5p) / (10u))) - 1) / 2.0) * ((2 * (60n > (((60n) - 0) + 50n) ? 60n : (((60n) - 0) + 50n))) + (2 * (({w}) / (ceil((({w}) - 5p) / (10u))))))) + ((((ceil((({w}) - 5p) / (10u))) / 2) - floor((ceil((({w}) - 5p) / (10u))) / 2) == 0) ? ((2 * (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))) + (2 * (({w}) / (ceil((({w}) - 5p) / (10u)))))) : 0)) / 1 \\
         pd=((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? ((floor((ceil((({w}) - 5p) / (10u))) / 2.0) * ((2 * (((60n) - 0) + 60n)) + 440n)) + ((((ceil((({w}) - 5p) / (10u))) / 2) - floor((ceil((({w}) - 5p) / (10u))) / 2) != 0) ? ((2 * (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n))) + 340n) : 0)) / 1 : ((floor((ceil((({w}) - 5p) / (10u))) / 2.0) * ((2 * (60n > (((60n) - 0) + 50n) ? 60n : (((60n) - 0) + 50n))) + (2 * (({w}) / (ceil((({w}) - 5p) / (10u))))))) + ((((ceil((({w}) - 5p) / (10u))) / 2) - floor((ceil((({w}) - 5p) / (10u))) / 2) != 0) ? ((2 * (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))) + (2 * (({w}) / (ceil((({w}) - 5p) / (10u)))))) : 0)) / 1 \\
         nrd=((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? ((floor((ceil((({w}) - 5p) / (10u))) / 2.0) * (((((60n) - 0) + 60n) * 120n) + ((({w}) / (ceil((({w}) - 5p) / (10u)))) * 100n))) + ((((ceil((({w}) - 5p) / (10u))) / 2) - floor((ceil((({w}) - 5p) / (10u))) / 2) != 0) ? (((50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) * 120n) + ((({w}) / (ceil((({w}) - 5p) / (10u)))) * 50n)) : 0)) / 1 : ((floor((ceil((({w}) - 5p) / (10u))) / 2.0) * ((60n > (((60n) - 0) + 50n) ? 60n : (((60n) - 0) + 50n)) * (({w}) / (ceil((({w}) - 5p) / (10u)))))) + ((((ceil((({w}) - 5p) / (10u))) / 2) - floor((ceil((({w}) - 5p) / (10u))) / 2) != 0) ? ((100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n)) * (({w}) / (ceil((({w}) - 5p) / (10u))))) : 0)) / 1 / ((({w}) / (ceil((({w}) - 5p) / (10u)))) * (ceil((({w}) - 5p) / (10u))) * (({w}) / (ceil((({w}) - 5p) / (10u)))) * (ceil((({w}) - 5p) / (10u)))) \\
         nrs=((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? ((((50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) * 120n) + ((({w}) / (ceil((({w}) - 5p) / (10u)))) * 50n)) + (floor(((ceil((({w}) - 5p) / (10u))) - 1) / 2.0) * (((((60n) - 0) + 60n) * 120n) + ((({w}) / (ceil((({w}) - 5p) / (10u)))) * 100n))) + ((((ceil((({w}) - 5p) / (10u))) / 2) - floor((ceil((({w}) - 5p) / (10u))) / 2) == 0) ? (((50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) * 120n) + ((({w}) / (ceil((({w}) - 5p) / (10u)))) * 50n)) : 0)) / 1 : (((100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n)) * (({w}) / (ceil((({w}) - 5p) / (10u))))) + (floor(((ceil((({w}) - 5p) / (10u))) - 1) / 2.0) * ((60n > (((60n) - 0) + 50n) ? 60n : (((60n) - 0) + 50n)) * (({w}) / (ceil((({w}) - 5p) / (10u)))))) + ((((ceil((({w}) - 5p) / (10u))) / 2) - floor((ceil((({w}) - 5p) / (10u))) / 2) == 0) ? ((100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n)) * (({w}) / (ceil((({w}) - 5p) / (10u))))) : 0)) / 1 / ((({w}) / (ceil((({w}) - 5p) / (10u)))) * (ceil((({w}) - 5p) / (10u))) * (({w}) / (ceil((({w}) - 5p) / (10u)))) * (ceil((({w}) - 5p) / (10u)))) \\
         sa=((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n)) \\
         sb=((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n)) \\
         sd=((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (((60n) - 0) + 60n) + (2*5e-08) : (60n > (((60n) - 0) + 50n) ? 60n : (((60n) - 0) + 50n)) \\
         sca=(( (({w}) / (ceil((({w}) - 5p) / (10u)))) * ( (((1u) * (1u) / (((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)) - ((1u) * (1u) / ((((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)+(45n)))) + ((1u) * (1u) / (((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)) - ((1u) * (1u)/ ((((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)+(45n))))) + ( (45n) * ( (((1u) * (1u) / (60n)) - ((1u) * (1u) / ((60n)+(({w}) / (ceil((({w}) - 5p) / (10u))))))) + ((1u) * (1u) / (60n)) - ((1u) * (1u)/ ((60n)+(({w}) / (ceil((({w}) - 5p) / (10u))))))))) / ((({w}) / (ceil((({w}) - 5p) / (10u)))) * (45n)) \\
         scb=(((({w}) / (ceil((({w}) - 5p) / (10u)))) * (((((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)/10 + (1u)/100)*exp(-10 * (((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n) / (1u)) - (((((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)+(45n))/10 + (1u)/100)*exp(-10 * ((((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)+(45n)) / (1u)) + ((((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)/10 + (1u)/100)*exp(-10 * (((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n) / (1u)) - (((((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)+(45n))/10 + (1u)/100)*exp(-10 * ((((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)+(45n)) / (1u)))) + ((45n) * (((60n)/10 + (1u)/100)*exp(-10 * (60n) / (1u)) - (((60n)+(({w}) / (ceil((({w}) - 5p) / (10u)))))/10 + (1u)/100)*exp(-10 * ((60n)+(({w}) / (ceil((({w}) - 5p) / (10u))))) / (1u)) + ((60n)/10 + (1u)/100)*exp(-10 * (60n) / (1u)) - (((60n)+(({w}) / (ceil((({w}) - 5p) / (10u)))))/10 + (1u)/100)*exp(-10 * ((60n)+(({w}) / (ceil((({w}) - 5p) / (10u))))) / (1u))))) / ((({w}) / (ceil((({w}) - 5p) / (10u)))) * (45n)) \\
         scc=(((({w}) / (ceil((({w}) - 5p) / (10u)))) * (((((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)/20 + (1u)/400)*exp(-20 * (((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n) / (1u)) - (((((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)+(45n))/20 + (1u)/400)*exp(-20 * ((((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)+(45n)) / (1u)) + ((((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)/20 + (1u)/400)*exp(-20 * (((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n) / (1u)) - (((((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)+(45n))/20 + (1u)/400)*exp(-20 * ((((({w}) / (ceil((({w}) - 5p) / (10u)))) < 119.5n) ? (50n > (((60n) - 0) + 60n) ? 50n : (((60n) - 0) + 60n)) + 5e-08 : (100n > (((60n) - 0) + 80n) ? 100n : (((60n) - 0) + 80n))+60n)+(45n)) / (1u)))) + ((45n) * (((60n)/20 + (1u)/400)*exp(-20 * (60n) / (1u)) - (((60n)+(({w}) / (ceil((({w}) - 5p) / (10u)))))/20 + (1u)/400)*exp(-20 * ((60n)+(({w}) / (ceil((({w}) - 5p) / (10u))))) / (1u)) + ((60n)/20 + (1u)/400)*exp(-20 * (60n) / (1u)) - (((60n)+(({w}) / (ceil((({w}) - 5p) / (10u)))))/20 + (1u)/400)*exp(-20 * ((60n)+(({w}) / (ceil((({w}) - 5p) / (10u))))) / (1u))))) / ((({w}) / (ceil((({w}) - 5p) / (10u)))) * (45n)) \\
         m=(1)'''

    for global_trans_name, data in master_netlist.items():
        inst_type = data['type']
        w_var = global_trans_name

        D = data['D']
        G = data['G']
        S = data['S']
        B = data['B']

        if inst_type == 'NMOS':
            decl = f"m_{global_trans_name} ({D} {G} {S} {B}) g45n1svt w=({w_var}) l=45n " + stress_block.replace("{w}", w_var)
            netlist_lines.append(decl)
        elif inst_type == 'PMOS':
            decl = f"m_{global_trans_name} ({D} {G} {S} {B}) g45p1svt w=({w_var}) l=45n " + stress_block.replace("{w}", w_var)
            netlist_lines.append(decl)

    for pin in output_pins:
        cap_var = f"cap_{pin}"
        cap_line = f"C_{pin} ({pin} 0) capacitor c=({cap_var})"
        netlist_lines.append(cap_line)

    netlist_string = "\n".join(netlist_lines)
    with open(filename, "w") as f:
        f.write(netlist_string)

# def get_heuristic_states(target_pin, input_pins, interconnections):
#     pin_states = {}
#
#     for pin in input_pins:
#         if pin == target_pin:
#             continue
#
#         nc_val = Non_controlling_default
#
#         for inst, data in interconnections.items():
#             raw_gate = data[0].upper()
#             connected_nets = [mapping[0] for mapping in data[1:]]
#
#             if pin in connected_nets:
#                 matched_gate = None
#                 for base_gate in NC_VALUES.keys():
#                     if raw_gate.startswith(base_gate):
#                         matched_gate = base_gate
#                         break
#
#                 nc_val = NC_VALUES.get(matched_gate, Non_controlling_default)
#                 break
#
#         pin_states[pin] = nc_val
#
#     return pin_states

# def append_stimuli_to_widths(width_filename, output_filename, input_pins, output_pins, interconnections, slew, cap, vdd_val):
#     with open(width_filename, 'r') as f:
#         header_line = f.readline().strip().replace('#', '').strip()
#     width_headers = header_line.split()
#     width_data = np.loadtxt(width_filename, skiprows=1)
#
#     if width_data.ndim == 1:
#         width_data = width_data.reshape(1, -1)
#
#     n_points = width_data.shape[0]
#     all_rows = []
#
#     new_headers = list(width_headers)
#     for pin in input_pins:
#         new_headers.extend([f"slew_{pin}", f"start_{pin}", f"end_{pin}"])
#     for pin in output_pins:
#         new_headers.append(f"cap_{pin}")
#
#     for target_pin in input_pins:
#         static_states = get_heuristic_states(target_pin, input_pins, interconnections)
#
#         for transition in [(0.0, vdd_val), (vdd_val, 0.0)]:
#             block = [width_data]
#
#             for pin in input_pins:
#                 slew_sec = slew.get(pin, 0.0) * 1e-12
#                 block.append(np.full((n_points, 1), slew_sec))
#
#                 if pin == target_pin:
#                     block.append(np.full((n_points, 1), transition[0]))
#                     block.append(np.full((n_points, 1), transition[1]))
#                 else:
#                     pin_volt = static_states[pin] * vdd_val
#                     block.append(np.full((n_points, 1), pin_volt))
#                     block.append(np.full((n_points, 1), pin_volt))
#
#             for pin in output_pins:
#                 cap_farads = cap * 1e-15
#                 block.append(np.full((n_points, 1), cap_farads))
#
#             all_rows.append(np.column_stack(block))
#
#     final_dataset = np.vstack(all_rows)
#     cadence_header = "simulator lang=spectre\nmy_dataset paramset {\n" + " ".join(new_headers)
#
#     np.savetxt(
#         output_filename,
#         final_dataset,
#         delimiter=" ",
#         header=cadence_header,
#         footer="}",
#         comments="",
#         fmt="%.6e"
#     )

# def generate_meas_scs(input_pins, output_pins, filename="meas.scs", vdd_val=1.0):
#     if not output_pins:
#         return
#
#     out_pin = output_pins[0]
#     thresh = vdd_val / 2.0
#
#     lines=["simulator lang=spice"]
#
#     for in_pin in input_pins:
#         lines.append(f".meas tran tpd_{in_pin}_rr trig v({in_pin}) val={thresh} rise=1 targ v({out_pin}) val={thresh} rise=1")
#         lines.append(f".meas tran tpd_{in_pin}_rf trig v({in_pin}) val={thresh} rise=1 targ v({out_pin}) val={thresh} fall=1")
#         lines.append(f".meas tran tpd_{in_pin}_fr trig v({in_pin}) val={thresh} fall=1 targ v({out_pin}) val={thresh} rise=1")
#         lines.append(f".meas tran tpd_{in_pin}_ff trig v({in_pin}) val={thresh} fall=1 targ v({out_pin}) val={thresh} fall=1")
#     with open(filename, "w") as f:
#         f.write("\n".join(lines))

# def generate_stimuli_scs(input_pins, filename="stimuli.scs", vdd_val=1.0):
#     lines = [
#         f"Vdd (VDD 0) vsource type=dc dc={vdd_val}",
#         "Vss (VSS 0) vsource type=dc dc=0",
#         ""
#     ]
#     for pin in input_pins:
#         lines.append(
#             f"V_{pin} ({pin} 0) vsource type=pulse "
#             f"val0=(start_{pin}) val1=(end_{pin}) "
#             f"delay=100p rise=(slew_{pin}) fall=(slew_{pin}) width=1n period=2n"
#         )
#     with open(filename, "w") as f:
#         f.write("\n".join(lines))

def append_stimuli_to_widths(width_filename, output_filename, input_pins, output_pins, slew, cap, vdd_val, target):
    with open(width_filename, 'r') as f:
        header_line = f.readline().strip().replace('#', '').strip()
    width_headers = header_line.split()
    width_data = np.loadtxt(width_filename, skiprows=1)

    if width_data.ndim == 1:
        width_data = width_data.reshape(1, -1)

    n_points = width_data.shape[0]

    new_headers = list(width_headers)
    for pin in input_pins:
        new_headers.extend([f"slew_{pin}", f"start_{pin}", f"end_{pin}"])
    for pin in output_pins:
        new_headers.append(f"cap_{pin}")

    # 1. Determine the transition based on the Target Path
    start_pin = target["start_pin"]
    if target["start_edge"] == "R":
        transition = (0.0, vdd_val) # Rising
    else:
        transition = (vdd_val, 0.0) # Falling

    # 2. Build the exact stimuli block
    block = [width_data]
    for pin in input_pins:
        slew_sec = slew.get(pin, 0.0) * 1e-12
        block.append(np.full((n_points, 1), slew_sec))

        if pin == start_pin:
            # Apply the dynamic pulse to the target pin
            block.append(np.full((n_points, 1), transition[0]))
            block.append(np.full((n_points, 1), transition[1]))
        else:
            # Apply the static side-input voltage (fallback to 1.0 if not specified)
            static_state = target["side_inputs"].get(pin, Non_controlling_default)
            pin_volt = static_state * vdd_val
            block.append(np.full((n_points, 1), pin_volt))
            block.append(np.full((n_points, 1), pin_volt))

    for pin in output_pins:
        cap_farads = cap[pin] * 1e-15
        block.append(np.full((n_points, 1), cap_farads))

    # 3. Save the single focused dataset
    final_dataset = np.column_stack(block)
    cadence_header = "simulator lang=spectre\nmy_dataset paramset {\n" + " ".join(new_headers)

    np.savetxt(
        output_filename,
        final_dataset,
        delimiter=" ",
        header=cadence_header,
        footer="}",
        comments="",
        fmt="%.6e"
    )

def generate_meas_scs(filename="meas.scs",config_filename = "config.scs", vdd_val=1.08, target=TARGET_PATH):
    thresh = 0.54

    start_pin = target["start_pin"]
    end_pin = target["end_pin"]

    # Translate 'R'/'F' into SPICE syntax
    trig_edge = "rise=1" if target["start_edge"] == "R" else "fall=1"
    targ_edge = "rise=1" if target["end_edge"] == "R" else "fall=1"

    lines = ["simulator lang=spice"]

    # Example label: tpd_A_rf (Rise at A, Fall at Sum)
    meas_label = f"tpd_{start_pin.lower()}_{target['start_edge'].lower()}{target['end_edge'].lower()}"

    # The single, precise measurement command
    meas_cmd = f".meas tran {meas_label} trig v({start_pin}) val={thresh} {trig_edge} targ v({end_pin}) val={thresh} {targ_edge}"
    lines.append(meas_cmd)

    with open(filename, "w") as f:
        f.write("\n".join(lines))

    with open(config_filename, 'r') as f:
        lines = f.readlines()
    idx = len(lines) - 1
    line = lines[idx]
    if line.startswith('save'):
        del lines[idx]

    new_line = f"save {start_pin} {end_pin}"
    lines.append(new_line)
    with open(config_filename, 'w') as f:
        f.write("".join(lines))

    return meas_label

def generate_stimuli_scs(input_pins, filename="stimuli.scs", vdd_val=1.08, target=TARGET_PATH):
    lines = [
        f"Vdd (VDD 0) vsource type=dc dc={vdd_val}",
        "Vss (VSS 0) vsource type=dc dc=0",
        ""
    ]

    start_pin = target["start_pin"]

    for pin in input_pins:
        if pin == start_pin:
            # The active pin driving the critical path gets the pulse
            lines.append(
                f"V_{pin} ({pin} 0) vsource type=pulse "
                f"val0=(start_{pin}) val1=(end_{pin}) "
                f"delay=500p rise=(slew_{pin}) fall=(slew_{pin})"
            )
        else:
            # The side-inputs get a highly efficient, flat DC source
            # It still reads from the paramset so it dynamically locks to 1.08V or 0V
            lines.append(
                f"V_{pin} ({pin} 0) vsource type=dc dc=(start_{pin})"
            )

    with open(filename, "w") as f:
        f.write("\n".join(lines))

def run_spectre(total_points):
    working_dir = os.getcwd()
    print(f"\n--- Launching Cadence Spectre (+APS 16-Core) ---")

    # 1. We wrap everything inside a bash execution string.
    # Bash sets the ulimit, THEN launches csh, THEN sources the Cadence env, THEN runs Spectre.
    # Notice the single quotes around localhost:4 to prevent string escaping nightmares.
    command = 'bash -c "ulimit -n 4096 && csh -c \\"source /cadence/cshrc ; spectre +aps +mt Spectre_SCS/top.scs\\""'

    try:
        # 2. shell=True is required to parse the complex chained bash command.
        # We REMOVED stdout=subprocess.PIPE. The output will print directly to your terminal.
        process = subprocess.Popen(
            command,
            cwd=working_dir,
            shell=True
        )

        print("[TELEMETRY] Starting Monte Carlo Sweep...")

        # 3. Now wait() is perfectly safe because Python isn't choking on a pipe buffer
        process.wait()

        if process.returncode == 0:
            print(f"\n--- Spectre execution successful! All {total_points} runs complete. ---")
        else:
            print(f"\n--- Spectre failed with return code {process.returncode} ---")

    except Exception as e:
        print(f"Failed to launch Spectre process: {e}")

def parse_and_preprocess_for_ml(mt0_filepath, dataset_filepath, output_csv, target, master_netlist, margin,width_range):
    print("\n--- Telemetry Extraction & ML Preprocessing ---")

    # 1. Parse the MT0 file (The y-Target)
    with open(mt0_filepath, 'r') as file:
        lines = file.readlines()

    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('index') or line.startswith('alter#'):
            start_idx = i
            break

    raw_text = " ".join(lines[start_idx:])
    tokens = raw_text.split()

    headers = []
    data_tokens = []
    is_header = True
    for t in tokens:
        if t == '1' and is_header:
            is_header = False
        if is_header:
            headers.append(t)
        else:
            data_tokens.append(t)

    num_cols = len(headers)
    data_matrix = np.array(data_tokens).reshape(-1, num_cols)
    df_meas = pd.DataFrame(data_matrix, columns=headers)

    meas_label = f"tpd_{target['start_pin'].lower()}_{target['start_edge'].lower()}{target['end_edge'].lower()}"
    if meas_label not in df_meas.columns:
        meas_label = [c for c in df_meas.columns if 'tpd' in c][0]

    df_meas[meas_label] = pd.to_numeric(df_meas[meas_label], errors='coerce')

    # 2. Parse the Dataset SCS (The X-Features)
    with open(dataset_filepath, 'r') as f:
        dataset_lines = f.readlines()

    data_start = 0
    dataset_headers = []
    for i, line in enumerate(dataset_lines):
        if line.startswith('my_dataset paramset'):
            dataset_headers = dataset_lines[i + 1].strip().split()
            data_start = i + 2
            break

    raw_feature_data = []
    for line in dataset_lines[data_start:]:
        if '}' in line:
            break
        raw_feature_data.append(line.strip().split())

    df_features = pd.DataFrame(raw_feature_data, columns=dataset_headers)
    df_features = df_features.apply(pd.to_numeric, errors='coerce')

    # =========================================================
    # 3. UNIT CONVERSION (SI to VLSI Units)
    # =========================================================
    for col in df_features.columns:
        if 'cap' in col:
            df_features[col] = (df_features[col] * 1e15).round(4)  # Farads to fF
        elif 'slew' in col:
            df_features[col] = (df_features[col] * 1e12).round(4)  # Seconds to ps
        elif col in master_netlist:
            df_features[col] = (df_features[col] * 1e6).round(4)  # Meters to um

    # 4. Stitch X and y Together
    min_rows = min(len(df_features), len(df_meas))
    df_features = df_features.iloc[:min_rows]
    df_meas = df_meas.iloc[:min_rows]

    df_final = pd.concat([df_features, df_meas[meas_label]], axis=1)
    df_final = df_final.dropna()

    # Target Unit Conversion (Seconds to ps)
    df_final[meas_label] = (df_final[meas_label] * 1e12).round(3)

    # 5. Zero-Variance Pruning
    feature_cols = [c for c in df_final.columns if c != meas_label]
    nunique = df_final[feature_cols].nunique()
    cols_to_drop = nunique[nunique == 1].index
    df_final = df_final.drop(columns=cols_to_drop)
    feature_cols = [c for c in feature_cols if c not in cols_to_drop]

    # =========================================================
    # 6. ABSOLUTE BOUNDS SCALING (Injecting physics into Scikit-Learn)
    # =========================================================
    scaler = MinMaxScaler(feature_range=(1, 2))

    data_min = []
    data_max = []

    for col in feature_cols:
        # Instead of recalculating, pull the exact min/max generated in step 1
        if col in width_range:
            min_w_meters, max_w_meters = width_range[col]

            # Convert the absolute meters bounds back to um to match the dataset
            theoretical_min = min_w_meters * 1e6
            theoretical_max = max_w_meters * 1e6

            data_min.append(theoretical_min)
            data_max.append(theoretical_max)
        else:
            # Fallback for Cap/Slew if you randomize them in the future
            data_min.append(df_final[col].min())
            data_max.append(df_final[col].max())

    # Hack the Scikit-Learn internal matrices
    data_min = np.array(data_min)
    data_max = np.array(data_max)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1.0  # Prevent zero-division

    scaler.data_min_ = data_min
    scaler.data_max_ = data_max
    scaler.data_range_ = data_range
    scaler.scale_ = (scaler.feature_range[1] - scaler.feature_range[0]) / data_range
    scaler.min_ = scaler.feature_range[0] - data_min * scaler.scale_
    scaler.n_features_in_ = len(feature_cols)
    scaler.feature_names_in_ = np.array(feature_cols)

    # Apply the forced transformation
    df_final[feature_cols] = scaler.transform(df_final[feature_cols])
    joblib.dump(scaler, 'master_scaler.pkl')
    

    # 7. Export
    df_final.to_csv(output_csv, index=False)
    print(f"\n[SUCCESS] Custom Bounded ML dataset generated: {output_csv}")

def Linear_optimizer_function(meas_label, simulation_result, master_netlist):
    df = pd.read_csv(simulation_result)

    # --- Data Cleaning ---
    original_count = len(df)
    df_filtered = df

    X = df_filtered.drop(columns=[meas_label])
    y = df_filtered[meas_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = joblib.load('master_scaler.pkl')

    print("--- Training Linear Regressor ---")
    lr_model = LinearRegression(fit_intercept=True)
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("\n--- Linear Regression Performance ---")
    print(f"R-squared (R2): {r2:.4f} (Closer to 1.0 is better)")
    print(f"Mean Absolute Error (MAE): {mae:.4e} (Raw physical error)")
    print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}% (Average % off)")

    def optimize_linear_surrogate(model, feature_names, scaler):
        print("\n--- Running PuLP Linear Optimization ---")
        lp_prob = pulp.LpProblem("Minimize_Circuit_Delay", pulp.LpMinimize)

        lp_vars = {
            name: pulp.LpVariable(name, lowBound=1.0, upBound=2.0)
            for name in feature_names
        }

        objective = model.intercept_
        for name, coef in zip(feature_names, model.coef_):
            objective += coef * lp_vars[name]

        lp_prob += objective
        status = lp_prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[status] == 'Optimal':
            min_delay = pulp.value(lp_prob.objective)
            print(f"[SUCCESS] Optimal Delay Found: {min_delay:.3f} ps")

            optimal_scaled = np.array([lp_vars[name].varValue for name in feature_names])
            optimal_um = scaler.inverse_transform(optimal_scaled.reshape(1, -1))[0]

            print("\n--- Optimal Transistor Sizing (um) ---")
            for name, val_um in zip(feature_names, optimal_um):
                print(f"{name}: {val_um:.6f} um")

            return dict(zip(feature_names, optimal_um))
        else:
            print(f"[FAILED] PuLP Status: {pulp.LpStatus[status]}")
            return None

    optimized_widths = optimize_linear_surrogate(lr_model, X_train.columns, scaler)

    # Stitch the full netlist back together for Spectre in METERS
    full_widths_meters = {}
    for name, data in master_netlist.items():
        if name in optimized_widths:
            # ML output is in micrometers (um). Convert to meters.
            full_widths_meters[name] = optimized_widths[name] * 1e-6
        else:
            # Pruned constant is natively in micrometers (um). Convert to meters.
            full_widths_meters[name] = float(data['instance'].W) * 1e-6

    return full_widths_meters

def Polynomial_optimizer_function(meas_label, simulation_result, master_netlist):
    df = pd.read_csv(simulation_result)

    # --- Data Cleaning ---
    original_count = len(df)
    df_filtered = df
    dropped_count = original_count - len(df_filtered)
    print(f"--- Data Cleaning ---")
    print(f"Original rows: {original_count}")
    print(f"Rows dropped (Delay > 245): {dropped_count}")
    print(f"Active training rows: {len(df_filtered)}\n")

    X = df_filtered.drop(columns=[meas_label])
    y = df_filtered[meas_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X_train, y_train)

    y_pred = poly_model.predict(X_test)
    print(f"--- Polynomial (Degree 3) Performance ---")
    print(f"R-squared (R2): {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred) * 100:.2f}%")

    y_train_pred = poly_model.predict(X_train)
    y_test_pred = poly_model.predict(X_test)
    print(f"Train R2: {r2_score(y_train, y_train_pred):.4f}")
    print(f"Test R2:  {r2_score(y_test, y_test_pred):.4f}")

    scaler = joblib.load('master_scaler.pkl')
    n_features = scaler.n_features_in_
    feature_names = scaler.feature_names_in_

    def surrogate_objective(x_scaled):
        df_x = pd.DataFrame([x_scaled], columns=feature_names)
        return poly_model.predict(df_x)[0]

    bounds = [(1.0, 2.0) for _ in range(n_features)]
    x0_scaled = np.full(n_features, 1.5)

    print("\n--- Running L-BFGS-B Circuit Optimization ---")
    result = minimize(
        surrogate_objective,
        x0_scaled,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 3000}
    )

    full_widths_um = {}
    full_widths_meters = {}
    if result.success:
        print(f"\n[SUCCESS] Optimal Delay Found: {result.fun:.3f} ps")
        optimal_widths_scaled = result.x.reshape(1, -1)
        optimal_widths_um = scaler.inverse_transform(optimal_widths_scaled)[0]

        print("\n--- Optimal Transistor Sizing (um) ---")
        for name, width in zip(feature_names, optimal_widths_um):
            print(f"{name}: {width:.6f} um")

        optimized_dict = dict(zip(feature_names, optimal_widths_um))
        # Stitch the full netlist back together for Spectre in METERS

        for name, data in master_netlist.items():
            if name in optimized_dict:
                # ML output is in micrometers (um). Convert to meters.
                full_widths_meters[name] = optimized_dict[name] * 1e-6
            else:
                # Pruned constant is natively in micrometers (um). Convert to meters.
                full_widths_meters[name] = float(data['instance'].W) * 1e-6


    else:
        print("\n[FAILED] Optimizer could not converge:", result.message)

    return full_widths_meters

def MLP_optimizer_function(meas_label, simulation_result, master_netlist):
    df = pd.read_csv(simulation_result)

    # --- Data Cleaning ---
    df_filtered = df

    X = df_filtered.drop(columns=[meas_label])
    y = df_filtered[meas_label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = joblib.load('master_scaler.pkl')
    feature_names = scaler.feature_names_in_

    print("--- Training Deep Surrogate (MLP) ---")
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(32, 16, 8),
        activation='logistic',
        solver='adam',
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        batch_size = 250
    )
    mlp_model.fit(X_train, y_train)

    y_pred = mlp_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("\n--- MLP Performance ---")
    print(f"R-squared (R2): {r2:.4f} (Closer to 1.0 is better)")
    print(f"Mean Absolute Error (MAE): {mae:.4e} (Raw physical error)")
    print(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}% (Average % off)")

    def mlp_objective(x_scaled):
        df_x = pd.DataFrame([x_scaled], columns=X_train.columns)
        return mlp_model.predict(df_x)[0]

    bounds = [(1.0, 2.0) for _ in range(len(X_train.columns))]
    x0_scaled = np.full(len(X_train.columns), 1.5)

    print("\n--- Running L-BFGS-B Circuit Optimization ---")
    result = minimize(
        mlp_objective,
        x0_scaled,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 3000}
    )

    full_widths_um = {}
    full_widths_meters = {}
    if result.success:
        print(f"\n[SUCCESS] Optimal Delay Found: {result.fun:.3f} ps")
        optimal_widths_scaled = result.x.reshape(1, -1)
        optimal_widths_um = scaler.inverse_transform(optimal_widths_scaled)[0]

        print("\n--- Optimal Transistor Sizing (um) ---")
        for name, width in zip(feature_names, optimal_widths_um):
            print(f"{name}: {width:.6f} um")

        optimized_dict = dict(zip(feature_names, optimal_widths_um))
        # Stitch the full netlist back together for Spectre in METERS

        for name, data in master_netlist.items():
            if name in optimized_dict:
                # ML output is in micrometers (um). Convert to meters.
                full_widths_meters[name] = optimized_dict[name] * 1e-6
            else:
                # Pruned constant is natively in micrometers (um). Convert to meters.
                full_widths_meters[name] = float(data['instance'].W) * 1e-6


    else:
        print("\n[FAILED] Optimizer could not converge:", result.message)

    return full_widths_meters

def generate_optimized_spectre_dataset(optimized_dicts, output_filename, input_pins, output_pins, slew, cap, vdd_val,
                                       target, Non_controlling_default=1.0):
    print(f"\n--- Generating Spectre Evaluation Dataset ---")

    # 1. Extract headers from the first dictionary
    width_headers = list(optimized_dicts[0].keys())
    n_points = len(optimized_dicts)

    # 2. Unpack the dictionaries and strip np.float64
    width_data_list = []
    for opt_dict in optimized_dicts:
        # Calling float() safely converts np.float64 back to a native Python float
        row = [float(opt_dict[key]) for key in width_headers]
        width_data_list.append(row)

    # Convert to a standard 2D numpy array
    width_data = np.array(width_data_list)

    # 3. Build the new headers including stimuli
    new_headers = list(width_headers)
    for pin in input_pins:
        new_headers.extend([f"slew_{pin}", f"start_{pin}", f"end_{pin}"])
    for pin in output_pins:
        new_headers.append(f"cap_{pin}")

    # 4. Determine the transition based on the Target Path
    start_pin = target["start_pin"]
    if target["start_edge"] == "R":
        transition = (0.0, vdd_val)  # Rising
    else:
        transition = (vdd_val, 0.0)  # Falling

    # 5. Build the exact stimuli block (matching your reference logic)
    block = [width_data]
    for pin in input_pins:
        slew_sec = slew.get(pin, 0.0) * 1e-12
        block.append(np.full((n_points, 1), slew_sec))

        if pin == start_pin:
            # Apply the dynamic pulse to the target pin
            block.append(np.full((n_points, 1), transition[0]))
            block.append(np.full((n_points, 1), transition[1]))
        else:
            # Apply the static side-input voltage
            static_state = target["side_inputs"].get(pin, Non_controlling_default)
            pin_volt = static_state * vdd_val
            block.append(np.full((n_points, 1), pin_volt))
            block.append(np.full((n_points, 1), pin_volt))

    for pin in output_pins:
        cap_farads = cap[pin] * 1e-15
        block.append(np.full((n_points, 1), cap_farads))

    # 6. Stack everything horizontally
    final_dataset = np.column_stack(block)

    # 7. Construct Cadence Header and Save
    cadence_header = "simulator lang=spectre\nmy_dataset paramset {\n" + " ".join(new_headers)

    np.savetxt(
        output_filename,
        final_dataset,
        delimiter=" ",
        header=cadence_header,
        footer="}",
        comments="",
        fmt="%.6e"
    )

    print(f"[SUCCESS] Exported {n_points} optimized points to {output_filename} ready for Spectre.")

def run_final_spectre(total_points):
    working_dir = os.getcwd()
    print(f"\n--- Launching Cadence Spectre (+APS 16-Core) ---")

    # 1. We wrap everything inside a bash execution string.
    # Bash sets the ulimit, THEN launches csh, THEN sources the Cadence env, THEN runs Spectre.
    # Notice the single quotes around localhost:4 to prevent string escaping nightmares.
    command = 'bash -c "ulimit -n 4096 && csh -c \\"source /cadence/cshrc ; spectre +aps Spectre_SCS/final.scs\\""'

    try:
        # 2. shell=True is required to parse the complex chained bash command.
        # We REMOVED stdout=subprocess.PIPE. The output will print directly to your terminal.
        process = subprocess.Popen(
            command,
            cwd=working_dir,
            shell=True
        )

        print("[TELEMETRY] Starting Monte Carlo Sweep...")

        # 3. Now wait() is perfectly safe because Python isn't choking on a pipe buffer
        process.wait()

        if process.returncode == 0:
            print(f"\n--- Spectre execution successful! All {total_points} runs complete. ---")
        else:
            print(f"\n--- Spectre failed with return code {process.returncode} ---")

    except Exception as e:
        print(f"Failed to launch Spectre process: {e}")


# --- MAIN EXECUTION FLOW ---
input_pins, output_pins, wire, interconnections = Verilogtraverser(netlist_filename)
dictionary = Netlist_Dictionary_Generator(input_pins, output_pins, wire, interconnections)
print("Extracted Dictionary:", dictionary)
# #
#n_points *= (len(input_pins) + len(output_pins) + len(dictionary))
n_points = 3
print("Total Monte Carlo Points:", n_points)
# #
# # # 1. Generate widths (input-facing transistors locked to constant values)
width_range = generate_ml_dataset(dictionary, input_pins, n_points, margin, input_dataset_filename)
# #
# # # 2. Append targeted stimuli (removed 'interconnections', added 'TARGET_PATH')
append_stimuli_to_widths(input_dataset_filename, input_dataset_filename, input_pins, output_pins, slew, cap, vdd, TARGET_PATH)
#
# 3. Generate the Spectre structural netlist
generate_scs_netlist(dictionary, input_pins, output_pins, spectre_netlist_filename)

# 4. Generate the targeted DC/Pulse stimuli (added 'TARGET_PATH')
generate_stimuli_scs(input_pins, spectre_stimuli_filename, vdd, TARGET_PATH)

# 5. Generate the single precise .meas statement (removed 'input/output_pins', added 'TARGET_PATH')
meas_label = generate_meas_scs(spectre_meas_filename,config_filename,vdd, TARGET_PATH)

run_spectre(n_points)
# 
# # Pass dictionary and margin to enforce theoretical Min/Max scaling
parse_and_preprocess_for_ml(
    meas_read_file,
    input_dataset_filename,
    simulation_result,
    TARGET_PATH,
    dictionary,
    margin,
    width_range
)

linear_output_dict = Linear_optimizer_function(meas_label, simulation_result,dictionary)
poly_output_dict = Polynomial_optimizer_function(meas_label, simulation_result,dictionary)
mlp_output_dict= MLP_optimizer_function(meas_label, simulation_result,dictionary)


# Put the 3 outputs from Linear, Poly, and MLP into a list
optimized_results = [
    linear_output_dict,
    poly_output_dict,
    mlp_output_dict
]

# Generate the .scs file
generate_optimized_spectre_dataset(
    optimized_dicts=optimized_results,
    output_filename=Finalresults_filename,
    input_pins=input_pins,
    output_pins=output_pins,
    slew=slew,
    cap=cap,
    vdd_val=vdd,
    target=TARGET_PATH
)

run_final_spectre(3)
