#DataXC.py
import json
import sys

current_timestep = 0
sizing_period_completed = False

def compute_new_setpoints(timestep, occupancy_status, zone_type, params):
    # Use params to set initial setpoints
    initial_heating_setpoint_guestroom = params["initial_heating_setpoint_guestroom"]
    initial_cooling_setpoint_guestroom = params["initial_cooling_setpoint_guestroom"]
    initial_heating_setpoint_corridor = params["initial_heating_setpoint_corridor"]
    initial_cooling_setpoint_corridor = params["initial_cooling_setpoint_corridor"]

    increment_value = 0.0

    if zone_type == "guestroom":
        if occupancy_status == "occupied":
            new_heating_setpoint = initial_heating_setpoint_guestroom + (increment_value * timestep)
            new_cooling_setpoint = initial_cooling_setpoint_guestroom + (increment_value * timestep)
        else:
            new_heating_setpoint = 18  # Lower heating setpoint when unoccupied
            new_cooling_setpoint = 28  # Higher cooling setpoint when unoccupied
    elif zone_type == "corridor":
        if occupancy_status == "occupied":
            new_heating_setpoint = initial_heating_setpoint_corridor + (increment_value * timestep)
            new_cooling_setpoint = initial_cooling_setpoint_corridor + (increment_value * timestep)
        else:
            new_heating_setpoint = 16  # Lower heating setpoint when unoccupied
            new_cooling_setpoint = 26  # Higher cooling setpoint when unoccupied

    return new_heating_setpoint, new_cooling_setpoint

def guestroom_occupancy_schedule(current_time):
    if 0 <= current_time < 8:
        return 1.0  # Fully occupied from midnight to 8:00 AM
    elif 8 <= current_time < 18:
        return 0.2  # Light occupancy during the day
    elif 18 <= current_time <= 24:
        return 0.8  # High occupancy in the evening
    else:
        return 0.0  # Fallback if current_time is outside 24-hour range

def corridor_occupancy_schedule(current_time):
    if 0 <= current_time < 6:
        return 0.0  # Unoccupied before 6:00 AM
    elif 6 <= current_time < 8:
        return 0.2  # Light occupancy in the early morning
    elif 8 <= current_time < 18:
        return 0.6  # Moderate occupancy during the day
    elif 18 <= current_time < 22:
        return 0.8  # High occupancy during peak hours in the evening
    elif 22 <= current_time <= 24:
        return 0.2  # Light occupancy late evening
    else:
        return 0.0  # Fallback if current_time is outside 24-hour range

def get_setpoints(occupancy_fraction, zone_type, params):
    global current_timestep, sizing_period_completed

    if sizing_period_completed:
        if occupancy_fraction > 0:
            heating_setpoint, cooling_setpoint = compute_new_setpoints(current_timestep, "occupied", zone_type, params)
        else:
            heating_setpoint, cooling_setpoint = compute_new_setpoints(current_timestep, "unoccupied", zone_type, params)
        current_timestep += 1
    else:
        if zone_type == "guestroom":
            heating_setpoint, cooling_setpoint = params["initial_heating_setpoint_guestroom"], params["initial_cooling_setpoint_guestroom"]
        elif zone_type == "corridor":
            heating_setpoint, cooling_setpoint = params["initial_heating_setpoint_corridor"], params["initial_cooling_setpoint_corridor"]
        current_timestep += 1
        if current_timestep >= 48:
            sizing_period_completed = True
            current_timestep = 0

    return heating_setpoint, cooling_setpoint

def get_infiltration_rates(params):
    infiltration_rate_guestrooms = params["infiltration_rate_guestrooms"]
    infiltration_rate_corridors = params["infiltration_rate_corridors"]
    return infiltration_rate_guestrooms, infiltration_rate_corridors

def get_occupancy_status(current_time, zone_type="guestroom"):
    if zone_type == "guestroom":
        occupancy_fraction = guestroom_occupancy_schedule(current_time)
    elif zone_type == "corridor":
        occupancy_fraction = corridor_occupancy_schedule(current_time)
    else:
        occupancy_fraction = 0.0

    return "occupied" if occupancy_fraction > 0.0 else "unoccupied"

def get_lighting_power_density(params, occupancy_fraction, zone_type):
    if zone_type == "guestroom":
        return params["lighting_power_density_guestroom"]
    elif zone_type == "corridor":
        return params["lighting_power_density_corridor"]
    return 0.0

def get_plug_load_power_density(params, occupancy_fraction, zone_type):
    if zone_type == "guestroom":
        return params["plug_load_power_density_guestroom"]
    elif zone_type == "corridor":
        return params["plug_load_power_density_corridor"]
    return 0.0

if __name__ == "__main__":
    # Load parameters from JSON file passed as argument
    json_filename = sys.argv[1]
    with open(json_filename, 'r') as f:
        params = json.load(f)

        current_time = 12  # Example current time, e.g., midday
        guestroom_occupancy_status = get_occupancy_status(current_time, "guestroom")
        corridor_occupancy_status = get_occupancy_status(current_time, "corridor")
        guestroom_occupancy_fraction = guestroom_occupancy_schedule(current_time)
        corridor_occupancy_fraction = corridor_occupancy_schedule(current_time)

        heating_setpoint_guestroom, cooling_setpoint_guestroom = get_setpoints(guestroom_occupancy_fraction, "guestroom", params)
        heating_setpoint_corridor, cooling_setpoint_corridor = get_setpoints(corridor_occupancy_fraction, "corridor", params)

        infiltration_rate_guestrooms, infiltration_rate_corridors = get_infiltration_rates(params)

        lighting_power_density_guestroom = get_lighting_power_density(params, guestroom_occupancy_fraction,"guestroom")
        lighting_power_density_corridor = get_lighting_power_density(params, corridor_occupancy_fraction, "corridor")

        plug_load_power_density_guestroom = get_plug_load_power_density(params, "guestroom")
        plug_load_power_density_corridor = get_plug_load_power_density(params, "corridor")

        output = {
            "heating_setpoint_guestroom": heating_setpoint_guestroom,
            "cooling_setpoint_guestroom": cooling_setpoint_guestroom,
            "heating_setpoint_corridor": heating_setpoint_corridor,
            "cooling_setpoint_corridor": cooling_setpoint_corridor,
            "infiltration_rate_guestrooms": infiltration_rate_guestrooms,
            "infiltration_rate_corridors": infiltration_rate_corridors,
            "guestroom_occupancy_status": guestroom_occupancy_status,
            "corridor_occupancy_status": corridor_occupancy_status,
            "guestroom_occupancy_fraction": guestroom_occupancy_fraction,
            "corridor_occupancy_fraction": corridor_occupancy_fraction,
            "lighting_power_density_guestroom": lighting_power_density_guestroom,
            "lighting_power_density_corridor": lighting_power_density_corridor,
            "plug_load_power_density_guestroom": plug_load_power_density_guestroom,
            "plug_load_power_density_corridor": plug_load_power_density_corridor
        }

        print(json.dumps(output, indent=4))