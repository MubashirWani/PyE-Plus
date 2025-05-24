#utils.py
import subprocess
import pandas as pd
import os
import json
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    return None

def write_json(data, file_path):
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(f"An unexpected error occurred while writing JSON: {e}")

# To clear the output directory after each iteration/simulation
def clear_output_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate over all the files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # Check if it's a file and remove it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # Check if it's a directory and remove it
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

# Clearing the eplusout.json file before beginning (Not needed for use)

# def clear_eplusout_file(output_dir):
#     eplusout_file = os.path.join(output_dir, 'eplusout.json')
#     try:
#         with open(eplusout_file, 'w') as f:
#             json.dump({}, f)
#         logging.info(f"Cleared contents of file: {eplusout_file}")
#     except Exception as e:
#         logging.error(f"An error occurred while clearing eplusout_measure.json: {e}")

def save_updated_training_data(X_train, Y_train, path='C:/Users/wanimh/PycharmProjects/pythonProject/.venv/Scripts/training_data_updated.json'):
    try:
        data_to_save = [
            {"params": np.array(x).tolist(), "objectives": list(y)}
            for x, y in zip(X_train, Y_train)
        ]
        with open(path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        print(f"[INFO] Updated training data saved to {path}")
    except Exception as e:
        print(f"[ERROR] Failed to save updated training data: {e}")

def run_energyplus(idf_path, output_dir, weather_file):
    command = [
        'energyplus',
        '-r',  # Run simulation
        '-w', weather_file,  # Weather file
        '-d', output_dir,  # Output directory
        idf_path  # IDF file
    ]

    # Uncomment the following lines to turn on EnergyPlus "RunPeriod" progress on console

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred while running EnergyPlus: {e}")

    # To suppress the EnergyPlus "RunPeriod" progress on console
    # try:
    #     with open(os.devnull, 'w') as devnull:
    #         subprocess.run(command, check=True, stdout=devnull, stderr=devnull)
    # except subprocess.CalledProcessError as e:
    #     logging.error(f"Error occurred while running EnergyPlus: {e}")


def save_results_to_json(output_dir, baseline_file, measure_file, data_type):

    data = pd.read_csv(os.path.join(output_dir, 'eplusout.csv'))

    # Initialize variables to store the sums
    total_energy_consumption_joules = 0
    gwp_natural_gas = 0
    gwp_electricity = 0
    joules_to_kwh = 2.77778e-7
    OccupantComfort_Guestroom = 0
    OccupantComfort_Corridor = 0

    if not data.empty:
        try:
            # Find the index of the columns with the headers "Electricity:Building J", "NaturalGasEmissions:CO2 kg", and "ElectricityEmissions:CO2 kg"
            energy_index = data.columns.get_loc("Electricity:Building [J](TimeStep)")
            gwp_ng_index = data.columns.get_loc("NaturalGasEmissions:CO2 [kg](TimeStep)")
            gwp_electricity_index = data.columns.get_loc("ElectricityEmissions:CO2 [kg](TimeStep)")
            OC_Guestroom_Index = data.columns.get_loc("THERMAL ZONE 1:Zone Air Temperature [C](TimeStep)")
            OC_Corridor_Index = data.columns.get_loc("THERMAL ZONE 2:Zone Air Temperature [C](TimeStep)")
            # OC_Corridor_Index = data.columns.get_loc("THERMAL ZONE 2:Zone Air Temperature [C](TimeStep)")

            # Iterate over each row in the DataFrame
            for i in range(0, 6552):
                total_energy_consumption_joules += float(data.iloc[i, energy_index])
                gwp_natural_gas += float(data.iloc[i, gwp_ng_index])
                gwp_electricity += float(data.iloc[i, gwp_electricity_index])
                OccupantComfort_Guestroom += float(data.iloc[i, OC_Guestroom_Index])
                OccupantComfort_Corridor += float(data.iloc[i, OC_Corridor_Index])

            # Convert total energy consumption from Joules to kWh
            total_energy_consumption_kwh = joules_to_kwh*total_energy_consumption_joules

            # Take average of Guestroom Temperatures
            OccupantComfort_Guestroom_Avg = OccupantComfort_Guestroom / 6552

            # Take average of Corridor Temperatures
            OccupantComfort_Corridor_Avg = OccupantComfort_Corridor/ 6552

            # Create a dictionary with both energy and GWP data
            result_data = {
                'total_energy_consumption': total_energy_consumption_kwh,
                'GWP_Natural_Gas': gwp_natural_gas,
                'GWP_Electricity': gwp_electricity,
                'Average_Guestroom_Temperature': OccupantComfort_Guestroom_Avg,
                'Average_Corridor_Temperature': OccupantComfort_Corridor_Avg
            }

            # Write baseline data only if the baseline file is specified
            if baseline_file and data_type == 'Baseline':
                write_json(result_data, baseline_file)
            # Write measure data only if the measure file is specified
            elif measure_file and data_type == 'Measure':
                write_json(result_data, measure_file)
            else:
                logging.error("Invalid data type or file path not provided.")

        except KeyError as e:
            logging.error(f"Key error: {e}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
    else:
        logging.error("The DataFrame is empty.")