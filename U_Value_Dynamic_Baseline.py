#U_Value_Dynamic_Baseline.py

# PyE+ - A Python-EnergyPlus Optimization Framework
# Copyright (c) 2025 Dr. Mubashir Hussain Wani
# Licensed under the MIT License. See LICENSE file in the project root for full license text.

from eppy import modeleditor
from eppy.modeleditor import IDF
import subprocess
import os
import json
import sys

# Load parameters from JSON file passed as argument
json_filename = sys.argv[1]
with open(json_filename, 'r') as f:
    params = json.load(f)

# Set the path to the IDD file
idd_path = "C:/EnergyPlusV23-2-0/Energy+.idd"
IDF.setiddname(idd_path)

# Load the IDF file
idf_path = "C:/Users/wanimh/PyCharmProjects/pythonProject/.venv/Scripts/Baseline_IDF_Modified.idf"
idf = IDF(idf_path)

# List all materials
materials = idf.idfobjects["MATERIAL"]
materials_no_mass = idf.idfobjects["MATERIAL:NOMASS"]

####### print("Materials:")
# for material in materials:
#     print(f"Name: {material.Name}, Conductivity: {material.Conductivity}, Thickness: {material.Thickness}")

####### ("\nMaterials (No Mass):")
# for material in materials_no_mass:
#     print(f"Name: {material.Name}, Thermal Resistance: {material.Thermal_Resistance}")

# Update properties for each material using parameters from Pilot script
####### print("Updating Materials:")
for material in idf.idfobjects["MATERIAL"]:
    material.Thermal_Absorptance = params["Thermal_Absorptance"]
    material.Solar_Absorptance = params["Solar_Absorptance"]
    material.Thickness = params["Thickness"]
    material.Conductivity = params["Conductivity"]
    ####### print(f"Updated {material.Name} with new properties")

####### print("\nUpdating Materials (No Mass):")
for material in idf.idfobjects["MATERIAL:NOMASS"]:
    material.Thermal_Absorptance = params["Thermal_Absorptance"]
    material.Solar_Absorptance = params["Solar_Absorptance"]
    # Note: Material:NoMass does not have Thickness and Conductivity properties
    ####### print(f"Updated {material.Name} with new properties")

# Function to calculate U-value
def calculate_u_value(material):
    if hasattr(material, 'Thickness') and hasattr(material, 'Conductivity'):
        return material.Conductivity / material.Thickness
    return None

# Calculate and print U-values for each material
####### print("\nCalculating U-values:")
for material in idf.idfobjects["MATERIAL"]:
    u_value = calculate_u_value(material)
    # if u_value is not None:
    #     print(f"Material: {material.Name}, U-value: {u_value:.4f} W/m²K")
    # else:
    #     print(f"Material: {material.Name} does not have Thickness or Conductivity defined")

# Save the modified IDF file with error handling
try:
    idf.save("Baseline_IDF_Modified.idf")
    print("Baseline IDF file updated successfully.")
except Exception as e:
    print(f"Error saving IDF file: {e}")




