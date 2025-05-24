#SixSigma.py
import json
import math
from pyenergyplus.plugin import EnergyPlusPlugin
from DataXC import get_setpoints, get_infiltration_rates, get_occupancy_status, guestroom_occupancy_schedule, corridor_occupancy_schedule, get_lighting_power_density, get_plug_load_power_density # Import the functions directly

class AllEnergyMeasures(EnergyPlusPlugin):
    ShadeStatusInteriorBlindOn = 6
    ShadeStatusOff = 0

    def __init__(self):
        super().__init__()
        self.handles_set = False
        self.heating_setpoint_handle_tz1 = None
        self.cooling_setpoint_handle_tz1 = None
        self.heating_setpoint_handle_tz2 = None
        self.cooling_setpoint_handle_tz2 = None
        self.lighting_power_density_handles = {}
        self.global_infiltration_handles = {}
        self.plug_load_global_handles = {}
        self.zone_floor_area_handles = {}
        self.lighting_global_handles = {}
        self.sub_surfaces = list(range(1, 46))  # Range Simplification
        self.solar_beam_incident_cosine_handles = {}
        self.shading_deploy_status_handles = {}
        self.global_shading_actuator_status_handles = {}
        self.plug_load_handles = {}
        self.infiltration_handles = {}
        self.occupancy_schedule_handles = {}

    def on_begin_timestep_before_predictor(self, state) -> int:
        if not self.handles_set:

            # Thermostat handles for Thermal Zone 1
            self.heating_setpoint_handle_tz1 = self.api.exchange.get_actuator_handle(
                state, "Schedule:Constant", "Schedule Value", "HTGSETP_SCH_TZ1"
            )
            self.cooling_setpoint_handle_tz1 = self.api.exchange.get_actuator_handle(
                state, "Schedule:Constant", "Schedule Value", "CLGSETP_SCH_TZ1"
            )

            # Thermostat handles for Thermal Zone 2
            self.heating_setpoint_handle_tz2 = self.api.exchange.get_actuator_handle(
                state, "Schedule:Constant", "Schedule Value", "HTGSETP_SCH_TZ2"
            )
            self.cooling_setpoint_handle_tz2 = self.api.exchange.get_actuator_handle(
                state, "Schedule:Constant", "Schedule Value", "CLGSETP_SCH_TZ2"
            )

            # Infiltration handles
            infiltration_names = [
                "Infiltration_GuestRooms",
                "Infiltration_Corridors"
            ]

            for name in infiltration_names:
                handle = self.api.exchange.get_actuator_handle(
                    state, "Zone Infiltration", "Air Exchange Flow Rate", name
                )
                self.infiltration_handles[name] = handle
                print(f"Infiltration Handle for {name}: {handle}")

            # Lighting power density handles
            lighting_names = [
                "GuestRooms_Lights",
                "Corridors_Lights"
            ]
            for name in lighting_names:
                handle = self.api.exchange.get_actuator_handle(
                    state, "Lights", "Electricity Rate", name
                )
                self.lighting_power_density_handles[name] = handle
                print(f"Lighting Handle for {name}: {handle}")

            # Occupancy schedule handles
            occupancy_names = [
                "Guestroom_Occupancy_Schedule",
                "Corridor_Occupancy_Schedule"
            ]
            for name in occupancy_names:
                handle = self.api.exchange.get_actuator_handle(
                    state, "Schedule:Compact", "Schedule Value", name
                )
                self.occupancy_schedule_handles[name] = handle
                print(f"Occupancy Schedule Handle for {name}: {handle}")

            # Zone floor area handles (TZ1 = 6000m2, TZ2 = 13440m2)
            zone_names = [
                "Thermal Zone 1",
                "Thermal Zone 2"
            ]
            for zone in zone_names:
                handle = self.api.exchange.get_internal_variable_handle(
                    state, "Zone Floor Area", zone
                )
                self.zone_floor_area_handles[zone] = handle
                print(f"Zone Floor Area Handle for {zone}: {handle}")

            # Global variable handles for lighting power density
            self.lighting_global_handles["GuestRooms"] = self.api.exchange.get_global_handle(
                state, "LightingPowerDensityGuestRooms"  # Global variable name
            )
            self.lighting_global_handles["Corridors"] = self.api.exchange.get_global_handle(
                state, "LightingPowerDensityCorridors"  # Global variable name
            )

            # Global variable handles for infiltration rates
            self.global_infiltration_handles["GuestRooms"] = self.api.exchange.get_global_handle(
                state, "InfiltrationRateGuestRooms"  # Global variable name
            )
            self.global_infiltration_handles["Corridors"] = self.api.exchange.get_global_handle(
                state, "InfiltrationRateCorridors"  # Global variable name
            )

            # Shade control handles for specified sub surfaces
            for i in self.sub_surfaces:
                self.solar_beam_incident_cosine_handles[i] = self.api.exchange.get_variable_handle(
                    state, "Surface Outside Face Beam Solar Incident Angle Cosine Value", f"Sub Surface {i}"
                )
                self.shading_deploy_status_handles[i] = self.api.exchange.get_actuator_handle(
                    state, "Window Shading Control", "Control Status", f"Sub Surface {i}"
                )
                self.global_shading_actuator_status_handles[i] = self.api.exchange.get_global_handle(
                    state, f"Sub_Surface_{i}_Shading_Deploy_Status"
                )

            # Plug load handles
            plug_load_names = [
                "GuestRooms_Plugs",
                "Corridors_Plugs"
            ]
            for name in plug_load_names:
                handle = self.api.exchange.get_actuator_handle(
                    state, "ElectricEquipment", "Electricity Rate", name
                )
                self.plug_load_handles[name] = handle
                print(f"Plug Load Handle for {name}: {handle}")

            # Global variable handles for plug load power density
            self.plug_load_global_handles["GuestRooms"] = self.api.exchange.get_global_handle(
                state, "PlugLoadPowerDensityGuestRooms"  # Global variable name
            )
            self.plug_load_global_handles["Corridors"] = self.api.exchange.get_global_handle(
                state, "PlugLoadPowerDensityCorridors"  # Global variable name
            )

            if (self.heating_setpoint_handle_tz1 == -1 or self.cooling_setpoint_handle_tz1 == -1 or
                    self.heating_setpoint_handle_tz2 == -1 or self.cooling_setpoint_handle_tz2 == -1 or
                    any(handle == -1 for handle in self.lighting_power_density_handles.values()) or
                    any(handle == -1 for handle in self.occupancy_schedule_handles.values()) or
                    any(handle == -1 for handle in self.zone_floor_area_handles.values()) or
                    any(handle == -1 for handle in self.lighting_global_handles.values()) or
                    any(handle == -1 for handle in self.solar_beam_incident_cosine_handles.values()) or
                    any(handle == -1 for handle in self.shading_deploy_status_handles.values()) or
                    any(handle == -1 for handle in self.global_shading_actuator_status_handles.values()) or
                    any(handle == -1 for handle in self.plug_load_handles.values()) or
                    any(handle == -1 for handle in self.infiltration_handles.values()) or
                    any(handle == -1 for handle in self.plug_load_global_handles.values())):
                self.api.runtime.issue_severe(state, "Could not get handle for one or more controls")
                return 1

            self.handles_set = True

        # Load parameters from JSON file
        with open('dataxc_input.json', 'r') as f:
            params = json.load(f)

        # Get current time from EnergyPlus
        # current_time = self.api.exchange.current_time(state)
        # occupancy_status = get_occupancy_status(current_time)

        current_time = self.api.exchange.hour(state)  # Get the hour of the day (0 to 24)
        # print(f"Current Time: {current_time}")
        occupancy_status = get_occupancy_status(current_time)

        guestroom_occupancy_status = get_occupancy_status(current_time, "guestroom")
        corridor_occupancy_status = get_occupancy_status(current_time, "corridor")

        guestroom_occupancy_fraction = guestroom_occupancy_schedule(current_time)
        corridor_occupancy_fraction = corridor_occupancy_schedule(current_time)

        # Thermostat control logic
        heating_setpoint_guestroom, cooling_setpoint_guestroom = get_setpoints(guestroom_occupancy_fraction, "guestroom", params)
        heating_setpoint_corridor, cooling_setpoint_corridor = get_setpoints(corridor_occupancy_fraction, "corridor", params)

        # heating_setpoint, cooling_setpoint = get_setpoints(occupancy_status) # Pass occupancy_status
        # self.api.exchange.set_actuator_value(state, self.heating_setpoint_handle, heating_setpoint)
        # self.api.exchange.set_actuator_value(state, self.cooling_setpoint_handle, cooling_setpoint)

        self._set_occupancy_schedule(state, "Guestroom_Occupancy_Schedule", guestroom_occupancy_status)
        self._set_occupancy_schedule(state, "Corridor_Occupancy_Schedule", corridor_occupancy_status)

        # Set thermostat setpoints for Thermal Zone 1
        self.api.exchange.set_actuator_value(state, self.heating_setpoint_handle_tz1, heating_setpoint_guestroom)
        self.api.exchange.set_actuator_value(state, self.cooling_setpoint_handle_tz1, cooling_setpoint_guestroom)

        # Set thermostat setpoints for Thermal Zone 2
        self.api.exchange.set_actuator_value(state, self.heating_setpoint_handle_tz2, heating_setpoint_corridor)
        self.api.exchange.set_actuator_value(state, self.cooling_setpoint_handle_tz2, cooling_setpoint_corridor)

        # Example infiltration control logic
        infiltration_rate_guestrooms, infiltration_rate_corridors = get_infiltration_rates(params)
        # print(f"Setting infiltration rate for GuestRooms: {infiltration_rate_guestrooms}")
        # print(f"Setting infiltration rate for Corridors: {infiltration_rate_corridors}")

        self.api.exchange.set_actuator_value(state, self.infiltration_handles["Infiltration_GuestRooms"], infiltration_rate_guestrooms)
        self.api.exchange.set_actuator_value(state, self.infiltration_handles["Infiltration_Corridors"], infiltration_rate_corridors)

        self._set_lighting_and_plug_loads(state, guestroom_occupancy_fraction, corridor_occupancy_fraction, params)

        # return heating_setpoint, cooling_setpoint

        occupancy_names = ["Guestroom_Occupancy_Schedule", "Corridor_Occupancy_Schedule"]
        for name in occupancy_names:
            handle = self.api.exchange.get_actuator_handle(
                state, "Schedule:Compact", "Schedule Value", name
            )
            self.occupancy_schedule_handles[name] = handle
        return 0
    def _set_occupancy_schedule(self, state, schedule_name, occupancy_status):
        value = 1.0 if occupancy_status == "occupied" else 0.0
        self.api.exchange.set_actuator_value(state, self.occupancy_schedule_handles[schedule_name], value)

    def _set_lighting_and_plug_loads(self, state, guestroom_occupancy_fraction, corridor_occupancy_fraction, params):
        # Example logic to set lighting power density and plug loads
        zone_floor_area_guestrooms = self.api.exchange.get_internal_variable_value(state, self.zone_floor_area_handles["Thermal Zone 1"])
        zone_floor_area_corridors = self.api.exchange.get_internal_variable_value(state, self.zone_floor_area_handles["Thermal Zone 2"])

        # new_lighting_power_density_guestrooms = get_lighting_power_density(params, guestroom_occupancy_fraction, "guestroom")
        # new_lighting_power_density_corridors = get_lighting_power_density(params, corridor_occupancy_fraction, "corridor")
        new_lighting_power_density_guestrooms = get_lighting_power_density(params, guestroom_occupancy_fraction, "guestroom")
        new_lighting_power_density_corridors = get_lighting_power_density(params, corridor_occupancy_fraction, "corridor")

        total_power_guestrooms = new_lighting_power_density_guestrooms * zone_floor_area_guestrooms
        total_power_corridors = new_lighting_power_density_corridors * zone_floor_area_corridors

        self.api.exchange.set_global_value(state, self.lighting_global_handles["GuestRooms"], new_lighting_power_density_guestrooms)
        self.api.exchange.set_global_value(state, self.lighting_global_handles["Corridors"], new_lighting_power_density_corridors)
        self.api.exchange.set_actuator_value(state, self.lighting_power_density_handles["GuestRooms_Lights"], total_power_guestrooms)
        self.api.exchange.set_actuator_value(state, self.lighting_power_density_handles["Corridors_Lights"], total_power_corridors)

        new_plug_load_guestrooms = get_plug_load_power_density(params, guestroom_occupancy_fraction, "guestroom")
        new_plug_load_corridors = get_plug_load_power_density(params, corridor_occupancy_fraction, "corridor")
        total_plug_load_guestrooms = new_plug_load_guestrooms * zone_floor_area_guestrooms
        total_plug_load_corridors = new_plug_load_corridors * zone_floor_area_corridors

        self.api.exchange.set_global_value(state, self.plug_load_global_handles["GuestRooms"], new_plug_load_guestrooms)
        self.api.exchange.set_global_value(state, self.plug_load_global_handles["Corridors"], new_plug_load_corridors)
        self.api.exchange.set_actuator_value(state, self.plug_load_handles["GuestRooms_Plugs"], total_plug_load_guestrooms)
        self.api.exchange.set_actuator_value(state, self.plug_load_handles["Corridors_Plugs"], total_plug_load_corridors)

        # Shade control logic for specified sub surfaces
        for i in self.sub_surfaces:
            current_incident_angle = self.api.exchange.get_variable_value(state, self.solar_beam_incident_cosine_handles[i])
            self._control_shade(state, current_incident_angle, self.shading_deploy_status_handles[i], self.global_shading_actuator_status_handles[i])

        return 0

    def _control_shade(self, state, current_incident_angle, shading_deploy_status_handle, global_shading_actuator_status_handle):
        # Ensure the cosine value is within the valid range
        if -1 <= current_incident_angle <= 1:
            incident_angle_rad = math.acos(current_incident_angle)
            incident_angle_degrees = math.degrees(incident_angle_rad)
            value_to_assign = self.ShadeStatusOff
            if incident_angle_degrees < 45:
                value_to_assign = self.ShadeStatusInteriorBlindOn
            self.api.exchange.set_actuator_value(state, shading_deploy_status_handle, value_to_assign)
            self.api.exchange.set_global_value(state, global_shading_actuator_status_handle, value_to_assign)
        else:
            self.api.runtime.issue_warning(state, "Invalid cosine value for incident angle: {}".format(current_incident_angle))