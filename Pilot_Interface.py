#!/usr/bin/env python
# Pilot_Interface.py
import sys
import numpy as np
from utils import read_json, write_json, run_energyplus, save_results_to_json, clear_output_folder
import os
import subprocess
import logging
import json
import argparse


overwrite = "True"

# Path to your existing baseline JSON
BASELINE_FILE = "C:/Users/wanimh/PyCharmProjects/pythonProject/.venv/Scripts/sim/output/energyplus_baseline.json"

def init_objective_ranges(baseline_file, comfort_bounds=(18.0, 28), T_ideal=23.0):
    # -------------------------
    # User-specified comfort setpoints and bounds
    # -------------------------
    T_ideal_guest = T_ideal
    T_ideal_corridor = T_ideal
    T_min_guest, T_max_guest = comfort_bounds
    T_min_corridor, T_max_corridor = comfort_bounds

    b = read_json(baseline_file)
    E_baseline = b['total_energy_consumption']
    GWP_base = b.get('GWP_Electricity', 0) + b.get('GWP_Natural_Gas', 0)

    T_ideal = np.array([T_ideal_guest, T_ideal_corridor])
    zone_bounds = np.array([(T_min_guest, T_max_guest),
                            (T_min_corridor, T_max_corridor)])

    # Worst-case scenario for each zone temperature (guestroom and corridor) - for OC only
    worst_guest = T_min_guest if abs(T_min_guest - T_ideal_guest) > abs(
        T_max_guest - T_ideal_guest) else T_max_guest
    worst_corridor = T_min_corridor if abs(T_min_corridor - T_ideal_corridor) > abs(
        T_max_corridor - T_ideal_corridor) else T_max_corridor
    T_worst = np.array([worst_guest, worst_corridor])

    # Euclidean Distance (maximum possible "comfort deviation" in the defined operating range)
    max_comfort_deviation = np.linalg.norm(T_worst - T_ideal)

    return {
        "energy_savings": {"min": 0.0, "max": E_baseline},
        "GWP_reduction": {"min": 0.0, "max": GWP_base},
        "OC": {"min": 0.0, "max": max_comfort_deviation},
    }

# objective_ranges = init_objective_ranges(BASELINE_FILE)

def normalize_objective(value, objective_name, epsilon=1e-6):
    global objective_ranges
    min_val = objective_ranges[objective_name]["min"]
    max_val = objective_ranges[objective_name]["max"]
    if (max_val - min_val) < epsilon:
        return 0.0 # Baseline (no gain for a maximization problem)

    if objective_name == "OC":
        if value <= max_val:
           normalized = (value - min_val) / (max_val - min_val)
           return 1 - normalized # Best is "1", worst (at max_val) is "0".
        else:
           # Smooth exponential decay beyond max_val
           decay_rate = 1.5 # Higher = faster penalty
           penalty = np.exp(-decay_rate * (value - max_val))
           return penalty * 0.01 # Keeping reward very small but > 0
    elif value < 0 and objective_name != "OC" and (objective_name == "energy_savings" or objective_name == "GWP_reduction"):
        k = abs(max_val) / 5 # Decay constant
        penalty = -(1 - np.exp(value / k))
        return float(penalty) # Decays smoothly to -1 (less negative lesser -ve penalty, more negative more -ve penalty)
    else:
        normalized = (value - min_val) / (max_val - min_val)
        return normalized

# -------------------------
# Main simulation functions
# -------------------------
def run_baseline_simulation(theta_baseline):
    baseline_idf_path = 'C:/Users/wanimh/PyCharmProjects/pythonProject/.venv/Scripts/Baseline_IDF_Modified.idf'
    output_dir = 'C:/Users/wanimh/PyCharmProjects/pythonProject/.venv/Scripts/sim/output'
    weather_file = 'C:/Users/wanimh/PyCharmProjects/pythonProject/.venv/Scripts/NZL_Wellington.Wellington.934360_IWEC.epw'
    os.makedirs(output_dir, exist_ok=True)
    baseline_file = os.path.join(output_dir, 'energyplus_baseline.json')
    measure_file = os.path.join(output_dir, 'energyplus_measure.json')
    clear_output_folder(output_dir)
    # Unpack theta_baseline values
    thermal_absorptance = float(theta_baseline[0])
    solar_absorptance = float(theta_baseline[1])
    thickness = float(theta_baseline[2])
    conductivity = float(theta_baseline[3])
    initial_heating_setpoint_guestroom = float(theta_baseline[4])
    initial_cooling_setpoint_guestroom = float(theta_baseline[5])
    initial_heating_setpoint_corridor = float(theta_baseline[6])
    initial_cooling_setpoint_corridor = float(theta_baseline[7])
    lighting_power_density_guestroom = float(theta_baseline[8])
    lighting_power_density_corridor = float(theta_baseline[9])
    plug_load_power_density_guestroom = float(theta_baseline[10])
    plug_load_power_density_corridor = float(theta_baseline[11])
    infiltration_rate_guestrooms = float(theta_baseline[12])
    infiltration_rate_corridors = float(theta_baseline[13])

    # Prepare JSON input for the U_Value_Dynamic script
    u_value_dynamic_input = {
        "Thermal_Absorptance": thermal_absorptance,
        "Solar_Absorptance": solar_absorptance,
        "Thickness": thickness,
        "Conductivity": conductivity,
        "initial_heating_setpoint_guestroom": initial_heating_setpoint_guestroom,
        "initial_cooling_setpoint_guestroom": initial_cooling_setpoint_guestroom,
        "initial_heating_setpoint_corridor": initial_heating_setpoint_corridor,
        "initial_cooling_setpoint_corridor": initial_cooling_setpoint_corridor,
        "lighting_power_density_guestroom": lighting_power_density_guestroom,
        "lighting_power_density_corridor": lighting_power_density_corridor,
        "plug_load_power_density_guestroom": plug_load_power_density_guestroom,
        "plug_load_power_density_corridor": plug_load_power_density_corridor,
        "infiltration_rate_guestrooms": infiltration_rate_guestrooms,
        "infiltration_rate_corridors": infiltration_rate_corridors
    }

    json_filename = 'dataxc_input.json'
    write_json(u_value_dynamic_input, json_filename)

    venv_activate = os.path.join('C:/Users/wanimh/PyCharmProjects/pythonProject/.venv/Scripts', 'activate.bat')
    u_value_dynamic_script_baseline = 'C:/Users/wanimh/PyCharmProjects/pythonProject/.venv/Scripts/U_Value_Dynamic_Baseline.py'

    try:
        subprocess.run(f'cmd /c "{venv_activate} && python {u_value_dynamic_script_baseline} {json_filename}"',
                       shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running U_Value_Dynamic_Baseline.py: {e}")

    if not os.path.exists(baseline_file) or overwrite == "True":
        print("Running baseline simulation.")
        run_energyplus(baseline_idf_path, output_dir, weather_file)
        save_results_to_json(output_dir, baseline_file, measure_file, 'Baseline')
    else:
        print("Baseline file already exists. Skipping baseline simulation.")

def co_simulate(theta_vect, comfort_bounds=(18.0, 28.0), T_ideal=23.0):

    global objective_ranges

    objective_ranges = init_objective_ranges(BASELINE_FILE, comfort_bounds, T_ideal)

    print("[DEBUG] Entered co_simulate with:", theta_vect)
    print("[DEBUG] theta_vect types:", [type(x) for x in theta_vect])

    energyplus_run_counter = get_run_counter()
    print(f"EnergyPlus Run # {energyplus_run_counter} for parameters: {theta_vect}")
    save_run_counter(energyplus_run_counter)

    thermal_absorptance = float(theta_vect[0])
    solar_absorptance = float(theta_vect[1])
    thickness = float(theta_vect[2])
    conductivity = float(theta_vect[3])
    initial_heating_setpoint_guestroom = float(theta_vect[4])
    initial_cooling_setpoint_guestroom = float(theta_vect[5])
    initial_heating_setpoint_corridor = float(theta_vect[6])
    initial_cooling_setpoint_corridor = float(theta_vect[7])
    lighting_power_density_guestroom = float(theta_vect[8])
    lighting_power_density_corridor = float(theta_vect[9])
    plug_load_power_density_guestroom = float(theta_vect[10])
    plug_load_power_density_corridor = float(theta_vect[11])
    infiltration_rate_guestrooms = float(theta_vect[12])
    infiltration_rate_corridors = float(theta_vect[13])

    u_value_dynamic_input = {
        "Thermal_Absorptance": thermal_absorptance,
        "Solar_Absorptance": solar_absorptance,
        "Thickness": thickness,
        "Conductivity": conductivity,
        "initial_heating_setpoint_guestroom": initial_heating_setpoint_guestroom,
        "initial_cooling_setpoint_guestroom": initial_cooling_setpoint_guestroom,
        "initial_heating_setpoint_corridor": initial_heating_setpoint_corridor,
        "initial_cooling_setpoint_corridor": initial_cooling_setpoint_corridor,
        "lighting_power_density_guestroom": lighting_power_density_guestroom,
        "lighting_power_density_corridor": lighting_power_density_corridor,
        "plug_load_power_density_guestroom": plug_load_power_density_guestroom,
        "plug_load_power_density_corridor": plug_load_power_density_corridor,
        "infiltration_rate_guestrooms": infiltration_rate_guestrooms,
        "infiltration_rate_corridors": infiltration_rate_corridors
    }

    json_filename = 'dataxc_input.json'
    write_json(u_value_dynamic_input, json_filename)

    idf_path = 'C:/Users/wanimh/PyCharmProjects/pythonProject/.venv/Scripts/RealBuilding_Test_MoreParameters_Wellington_Python.idf'
    output_dir = 'C:/Users/wanimh/PyCharmProjects/pythonProject/.venv/Scripts/sim/output'
    weather_file = 'C:/Users/wanimh/PyCharmProjects/pythonProject/.venv/Scripts/NZL_Wellington.Wellington.934360_IWEC.epw'
    os.makedirs(output_dir, exist_ok=True)
    baseline_file = os.path.join(output_dir, 'energyplus_baseline.json')
    measure_file = os.path.join(output_dir, 'energyplus_measure.json')

    # CLEAR OLD MEASURE RESULTS
    try:
        if os.path.exists(measure_file):
            os.remove(measure_file)
            print(f"Removed stale measure results")
    except OSError as e:
        print(f"Warning: Could not remove old measure file")

    venv_activate = os.path.join('C:/Users/wanimh/PyCharmProjects/pythonProject/.venv/Scripts', 'activate.bat')
    u_value_dynamic_script_measure = 'C:/Users/wanimh/PyCharmProjects/pythonProject/.venv/Scripts/U_Value_Dynamic_Measure.py'

    # try:
    #     subprocess.run(f'cmd /c "{venv_activate} && python {u_value_dynamic_script_measure} {json_filename}"',
    #                    shell=True, check=True)
    # except subprocess.CalledProcessError as e:
    #     print(f"Error occurred while running U_Value_Dynamic_Measure.py: {e}")
    #
    # run_energyplus(idf_path, output_dir, weather_file)
    # save_results_to_json(output_dir, baseline_file, measure_file, 'Measure')

    try:
        # patch the Measure IDF
        subprocess.run(
            f'cmd /c "{venv_activate} && python {u_value_dynamic_script_measure} {json_filename}"',
            shell=True, check=True
        )
        # run the full simulation
        run_energyplus(idf_path, output_dir, weather_file)
        # collect results
        save_results_to_json(output_dir, baseline_file, measure_file, 'Measure')

    # except Exception as e:
    #     # log and re-throw so NSGA-II’s evaluate() can catch it
    #     print(f"[co_simulate ERROR] Simulation failed for parameters {theta_vect}: {e}")
    #     raise
    except Exception as e:
    # Catch the E+ return-code failure an dbubble it up
        print(f"[co_simulate ERROR] Simulation failed for parameters {theta_vect}: {e}")
        raise

    if not os.path.exists(measure_file):
        raise RuntimeError(f"Measure JSON not created for parameters {theta_vect}")
    measure_data = read_json(measure_file)
    if measure_data is None:
        raise RuntimeError(f"Measure JSON invalid for parameters {theta_vect}")

    def calculate_energy_savings():
        try:
            baseline_data = read_json(baseline_file)
            measure_data = read_json(measure_file)
            print("Baseline Data:", baseline_data)
            print("Measure Data:", measure_data)
            if baseline_data and measure_data:
                E_baseline = baseline_data['total_energy_consumption']
                E_measure = measure_data['total_energy_consumption']
                energy_savings = E_baseline - E_measure
                if energy_savings is not None:
                    print(f"Energy Savings: {energy_savings} kWh/year")
                    #update_objective_ranges("energy_savings", energy_savings)
                    normalized_energy_savings = normalize_objective(energy_savings, "energy_savings")
                    return normalized_energy_savings
            return None
        except (KeyError, TypeError) as e:
            print(f"Error calculating energy savings: {e}")
            return None

    energy_savings = calculate_energy_savings()
    if energy_savings is not None:
        print(f"Normalized Energy Savings: {energy_savings}")
    else:
        print("Error calculating energy savings.")

    def retrieve_gwp_data():
        try:
            baseline_data = read_json(baseline_file)
            measure_data = read_json(measure_file)
            print("Baseline Data:", baseline_data)
            print("Measure Data:", measure_data)
            if baseline_data and measure_data:
                GWP_baseline_electricity = baseline_data.get('GWP_Electricity', 0)
                GWP_baseline_natural_gas = baseline_data.get('GWP_Natural_Gas', 0)
                GWP_measure_electricity = measure_data.get('GWP_Electricity', 0)
                GWP_measure_natural_gas = measure_data.get('GWP_Natural_Gas', 0)
                GWP_base = (GWP_baseline_electricity + GWP_baseline_natural_gas)
                GWP_reduction = ((GWP_baseline_electricity + GWP_baseline_natural_gas) - (
                            GWP_measure_electricity + GWP_measure_natural_gas))
                if GWP_reduction is not None:
                    print(f"GWP Reduction: {GWP_reduction} kg CO2e/year")
                    #update_objective_ranges("GWP_reduction", GWP_reduction)
                    normalized_gwp_reduction = normalize_objective(GWP_reduction, "GWP_reduction")
                    return normalized_gwp_reduction
            return None
        except (KeyError, TypeError) as e:
            print(f"Error retrieving GWP data: {e}")
            return None

    GWP_reduction = retrieve_gwp_data()
    if GWP_reduction is not None:
        print(f"Normalized GWP Reduction: {GWP_reduction}")
    else:
        print("Error retrieving GWP data.")

    def retrieve_occupantcomfort_data():

        T_IDEAL = np.array([T_ideal, T_ideal])

        try:
            baseline_data = read_json(baseline_file)
            measure_data = read_json(measure_file)
            print("Baseline Data:", baseline_data)
            print("Measure Data:", measure_data)
            if baseline_data and measure_data:
                Average_baseline_Temp_Guestroom = baseline_data.get('Average_Guestroom_Temperature')
                Average_measure_Temp_Guestroom = measure_data.get('Average_Guestroom_Temperature')
                Average_baseline_Temp_Corridor = baseline_data.get('Average_Corridor_Temperature')
                Average_measure_Temp_Corridor = measure_data.get('Average_Corridor_Temperature')
                # OC = 0.5 * (abs((T_ideal - Average_measure_Temp_Guestroom) / Average_baseline_Temp_Guestroom)) + \
                #      0.5 * (abs((T_ideal - Average_measure_Temp_Corridor) / Average_baseline_Temp_Corridor))
                # OC = np.sqrt((T_ideal - Average_measure_Temp_Guestroom)**2 + (T_ideal - Average_measure_Temp_Corridor)**2)
                # OC = 1 - OC
                actual = np.array([
                    Average_measure_Temp_Guestroom,
                    Average_measure_Temp_Corridor
                ])

                # Euclidean Distance - from the ideal
                dev = np.linalg.norm(actual - T_IDEAL)
                OC = normalize_objective(dev, "OC")
                #OC = (dev / max_comfort_deviation)

                print(f"Max. Comfort Deviation (Worst Case): {objective_ranges['OC']['max']} °C")

                if dev is not None:
                    print(f"Occupant Comfort (Deviation from Ideal): {dev} °C")
                    return OC
            return None
        except (KeyError, TypeError) as e:
            print(f"Error retrieving occupant comfort data: {e}")
            return None

    OC = retrieve_occupantcomfort_data()
    if OC is not None:
        print(f"Normalized Occupant Comfort: {OC}")
    else:
        print("Error retrieving occupant comfort data.")


    objectives = [energy_savings, GWP_reduction, OC]
    print("Objectives:", json.dumps(objectives))
    return objectives


def get_run_counter():
    try:
        with open("run_counter.txt", "r") as f:
            return int(f.read()) + 1
    except Exception:
        return 1


def save_run_counter(counter):
    # Open in write mode so that the counter can be updated
    with open("run_counter.txt", "w") as f:
        f.write(str(counter))
