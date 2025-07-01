from __future__ import annotations

import time
from typing import cast

import numpy as np

import porepy as pp

# geometry description horizontal case
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (  # noqa: E501
    SimpleGeometryHorizontal as ModelGeometryH,
)
from porepy.examples.geothermal_flow.model_configuration.geometry_description.geometry_market import (  # noqa: E501
    SimpleGeometryVertical as ModelGeometryV,
)


# Figure 5 two with low pressure (lP) condition
# Horizontal without gravity
# Vertical with gravity

from porepy.examples.geothermal_flow.model_configuration.DriesnerModelConfiguration import (  # noqa: E501
    DriesnerBrineFlowModel as FlowModel,
)

from porepy.examples.geothermal_flow.model_configuration.bc_description.bc_market import (  # noqa: E501
    BC_two_phase_moderate_pressure as BC,
)

from porepy.examples.geothermal_flow.model_configuration.ic_description.ic_market import (  # noqa: E501
    IC_two_phase_moderate_pressure as IC,
)
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

# Main directives
case_name = "case_lP"
geometry_case = "horizontal"

final_times = {
    "horizontal": [73000.0],  # final time [200 years]
    "vertical": [365000.0],  # final time [1000 years]
}

day_to_second = 86400
to_Mega = 1.0e-6
# Configuration dictionary mapping cases to their specific classes
simulation_cases = {
    "case_lP": {
        "tf": final_times[geometry_case][0] * day_to_second,  # final time [years]
        "dt": 0.5 * 365.0 * day_to_second,  # final time [1 years]
        "bc": BC,
        "ic": IC,
    }
}

geometry_cases = {
    "horizontal": ModelGeometryH,
    "vertical": ModelGeometryV,
}

tf = cast(float, simulation_cases[case_name]["tf"])
dt = cast(float, simulation_cases[case_name]["dt"])
BoundaryConditions: type = cast(type, simulation_cases[case_name]["bc"])
InitialConditions: type = cast(type, simulation_cases[case_name]["ic"])
ModelGeometry: type = cast(type, geometry_cases[geometry_case])

time_manager = pp.TimeManager(
    schedule=[0.0, tf],
    dt_init=dt,
    constant_dt=True,
    iter_max=50,
    print_info=True,
)

solid_constants = pp.SolidConstants(
    permeability=1e-15,
    porosity=0.1,
    thermal_conductivity=2.0 * to_Mega,
    density=2700.0,
    specific_heat_capacity=880.0 * to_Mega,
)
material_constants = {"solid": solid_constants}
params = {
    "material_constants": material_constants,
    "fractional_flow": False,
    "buoyancy_on": False,
    "time_manager": time_manager,
    "prepare_simulation": False,
    "apply_schur_complement_reduction": False,
    "nl_convergence_tol": np.inf,
    "nl_convergence_tol_res": 1.0e-2,
    "max_iterations": 200,
}


class GeothermalWaterFlowModel(
    ModelGeometry, BoundaryConditions, InitialConditions, FlowModel
):
    def after_nonlinear_convergence(self) -> None:
        second_to_year = 1.0 / (365 * day_to_second)
        super().after_nonlinear_convergence()  # type:ignore[safe-super]
        print("Number of iterations: ", self.nonlinear_solver_statistics.num_iteration)
        print("Time value (year): ", self.time_manager.time * second_to_year)
        print("Time index: ", self.time_manager.time_index)
        print("")

    def gravity_field(self, subdomains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
        g_constant = pp.GRAVITY_ACCELERATION
        val = self.units.convert_units(g_constant, "m*s^-2") * to_Mega
        size = np.sum([g.num_cells for g in subdomains]).astype(int)
        gravity_field = pp.wrap_as_dense_ad_array(val, size=size)
        gravity_field.set_name("gravity_field")
        return gravity_field


# Instance of the computational model
model = GeothermalWaterFlowModel(params)

parametric_space_ref_level = 2
folder_prefix = "src/porepy/examples/geothermal_flow/"
file_name_prefix = (
    "model_configuration/constitutive_description/driesner_vtk_files/"
)
file_name_phz = (
    file_name_prefix
    + "XHP_l"
    + str(parametric_space_ref_level)
    + "_modified_low_salt_content.vtk"
)
file_name_ptz = (
    file_name_prefix
    + "XTP_l"
    + str(parametric_space_ref_level)
    + "_modified_low_salt_content.vtk"
)

brine_sampler_phz = VTKSampler(file_name_phz)
brine_sampler_phz.conversion_factors = (1.0, 1.0e3, 10.0)  # (z,h,p)
model.vtk_sampler = brine_sampler_phz

brine_sampler_ptz = VTKSampler(file_name_ptz)
brine_sampler_ptz.conversion_factors = (1.0, 1.0, 10.0)  # (z,t,p)
brine_sampler_ptz.translation_factors = (0.0, -273.15, 0.0)  # (z,t,p)
model.vtk_sampler_ptz = brine_sampler_ptz


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# --- 0. Placeholder Data Generation ---
# This part creates dummy CSV files for demonstration purposes.
# In your actual use case, you can remove this section and use your real files.
print("Creating placeholder data files...")

# Create some non-linear, sparse data to make interpolation meaningful
distance_sparse = np.array([0.0, 0.5, 0.8, 1.1, 1.15, 1.2, 1.5, 2.0])

# Saturation Data
sat_liq_sparse = np.array([1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.6, 0.5])
df_sat_liq = pd.DataFrame({'Distance (km)': distance_sparse, 'Liquid Saturation': sat_liq_sparse})
df_sat_liq.to_csv('fig_5_vertical_saturation_liq_raw.csv', index=False)

print("Placeholder files created.")
# --- End of Placeholder Section ---


# --- 1. Load Data and Create Interpolators ---
print("\nStep 1: Loading data and creating interpolators...")

# Dictionary to hold data and interpolators
data_fields = {
    'Temperature': {'file': 'fig_5_vertical_temperature_raw.csv', 'column': 'Temperature (°C)'},
    'Pressure': {'file': 'fig_5_vertical_pressured_raw.csv', 'column': 'Pressure (MPa)'},
    'Saturation': {'file': 'fig_5_vertical_saturation_liq_raw.csv', 'column': 'Liquid Saturation'}
}

interpolators = {}

for key, field in data_fields.items():
    try:
        # Load the raw data
        df_raw = pd.read_csv(field['file'])

        # Extract x and y values for interpolation
        x_raw = df_raw['Distance (km)']
        y_raw = df_raw[field['column']]

        # Create a 1D linear interpolator function
        # fill_value="extrapolate" allows evaluation outside the original data range
        interpolators[key] = interp1d(x_raw, y_raw, kind='linear', fill_value="extrapolate")
        print(f"  - Interpolator for '{key}' created successfully.")

    except FileNotFoundError:
        print(f"Error: The file '{field['file']}' was not found.")
        exit()

# --- 2. Define New Grid and Compute Interpolated Values ---
print("\nStep 2: Resampling data on a regular grid...")

# Define a regular distance grid with 0.1 km resolution
resolution = 0.025
distance_regular = np.arange(0.0, 2.0 + resolution, resolution)

# Create a new DataFrame to hold the interpolated results
df_interpolated = pd.DataFrame({'Distance (km)': distance_regular})

# Compute the interpolated value for each field at the new distance points
df_interpolated['Temperature (°C)'] = interpolators['Temperature'](distance_regular)
df_interpolated['Pressure (MPa)'] = interpolators['Pressure'](distance_regular)
df_interpolated['Liquid Saturation'] = interpolators['Saturation'](distance_regular)

print("Interpolated DataFrame head:")
print(df_interpolated.head())

# --- 3. Plot the Interpolated Quantities ---
print("\nStep 3: Generating plots...")

# Create a figure with two subplots stacked vertically, sharing the x-axis
fig, (ax1, ax2) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 10),
    sharex=True  # Both subplots will share the same x-axis
)
fig.suptitle('Interpolated Geothermal Properties vs. Distance', fontsize=16)

# --- Subplot 1: Temperature and Pressure ---
ax1_twin = ax1.twinx()  # Create a secondary y-axis

# Plot Temperature on the primary y-axis (left)
color_temp = 'red'
ax1.set_ylabel('Temperature (°C)', color=color_temp)
ax1.plot(df_interpolated['Distance (km)'], df_interpolated['Temperature (°C)'], color=color_temp, marker='o',
         linestyle='-', label='Temperature')
ax1.tick_params(axis='y', labelcolor=color_temp)
ax1.grid(True, linestyle=':')

# Plot Pressure on the secondary y-axis (right)
color_pressure = 'blue'
ax1_twin.set_ylabel('Pressure (MPa)', color=color_pressure)
ax1_twin.plot(df_interpolated['Distance (km)'], df_interpolated['Pressure (MPa)'], color=color_pressure, marker='s',
              linestyle='--', label='Pressure')
ax1_twin.tick_params(axis='y', labelcolor=color_pressure)

# --- Subplot 2: Liquid Saturation ---



# compute saturation
tref_v = df_interpolated['Temperature (°C)'].values + 273.15
p_v = df_interpolated['Pressure (MPa)'].values


def find_properties_by_enthalpy(tref_v, p_v, model, h_min=0.1, h_max=3.5, max_iterations=100, tolerance=1e-3):

    print("--- Starting Bisection Search for Enthalpy ---")
    num_points = len(tref_v)

    # Initialize lower (ha_v) and upper (hb_v) enthalpy bounds for all points
    ha_v = h_min * np.ones(num_points)
    hb_v = h_max * np.ones(num_points)

    for i in range(max_iterations):
        # Calculate the midpoint enthalpy for the current interval for all points
        h_mid_v = 0.5 * (ha_v + hb_v)

        # Prepare the input points for the model sampler
        par_points = np.array((np.zeros(num_points), h_mid_v, p_v)).T

        # Call the model to get the temperature based on the current enthalpy guess
        model.vtk_sampler.sample_at(par_points)
        t_calc_v = model.vtk_sampler.sampled_could.point_data["Temperature"]

        # Calculate the difference between reference and calculated temperatures
        t_diff_v = tref_v - t_calc_v

        # Check if all points have converged
        if np.all(np.abs(t_diff_v) < tolerance):
            print(f"Convergence reached for all points at iteration {i + 1}.")
            break

        # Update the bounds for the next iteration using NumPy's `where` for efficiency.
        # If t_diff > 0, t_calc is too low, so we need higher enthalpy -> new lower bound is h_mid
        # If t_diff <= 0, t_calc is too high, so we need lower enthalpy -> new upper bound is h_mid
        ha_v = np.where(t_diff_v > 0, h_mid_v, ha_v)
        hb_v = np.where(t_diff_v <= 0, h_mid_v, hb_v)

        print(f"  Iteration {i + 1:2d}/{max_iterations}: Max |ΔT| = {np.max(np.abs(t_diff_v)):.4f} K")

    else:  # This 'else' belongs to the 'for' loop and runs if the loop completes without 'break'
        print("Warning: Maximum iterations reached without convergence for all points.")

    # The final converged enthalpy is the last midpoint
    h_final_v = 0.5 * (ha_v + hb_v)

    # Perform one final sampling with the converged enthalpy to get all final properties
    print("Performing final sampling with converged enthalpy values...")
    final_par_points = np.array((np.zeros(num_points), h_final_v, p_v)).T
    model.vtk_sampler.sample_at(final_par_points)

    t_final_v = model.vtk_sampler.sampled_could.point_data["Temperature"]
    s_gas_final_v = model.vtk_sampler.sampled_could.point_data["S_v"]

    print("--- Bisection Search Complete ---")

    return t_final_v, h_final_v, s_gas_final_v

t_final_v, h_final_v, s_gas_final_v = find_properties_by_enthalpy(tref_v, p_v, model)
s_liq = 1.0 - s_gas_final_v

color_sat = 'green'
ax2.set_ylabel('Liquid Saturation', color=color_sat)
ax2.plot(df_interpolated['Distance (km)'], s_liq, color=color_sat, marker='^',
         linestyle='-', label='Liquid Saturation')
ax2.tick_params(axis='y', labelcolor=color_sat)
ax2.set_ylim(0, 1.1)
ax2.grid(True, linestyle=':')

# --- Final Touches ---
# Set the shared x-axis label on the bottom plot
ax2.set_xlabel('Distance (km)', fontsize=12)

# Create a unified legend for the first plot
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Set legend for the second plot
ax2.legend(loc='upper right')

# Adjust layout and save the figure
fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
plt.savefig('reproduced_plot.png')
print("\nPlot saved as 'interpolated_properties_plot.png'")

tb = time.time()
model.prepare_simulation()
te = time.time()
print("Elapsed time prepare simulation: ", te - tb)
print("Simulation prepared for total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid employed: ", model.mdg)
model.schur_complement_primary_equations = (
    pp.compositional_flow.get_primary_equations_cf(model)
)
model.schur_complement_primary_variables = (
    pp.compositional_flow.get_primary_variables_cf(model)
)

# print geometry
model.exporter.write_vtu()
tb = time.time()
pp.run_time_dependent_model(model, params)
te = time.time()
print("Elapsed time run_time_dependent_model: ", te - tb)
print("Total number of DoF: ", model.equation_system.num_dofs())
print("Mixed-dimensional grid information: ", model.mdg)

# Retrieve the grid and boundary information
grid = model.mdg.subdomains()[0]
bc_sides = model.domain_boundary_sides(grid)

# Integrated overall mass flux on all facets
mn = model.equation_system.evaluate(model.darcy_flux(model.mdg.subdomains()))
mn = cast(np.ndarray, mn)

inlet_idx, outlet_idx = model.get_inlet_outlet_sides(model.mdg.subdomains()[0])
print("Inflow values : ", mn[inlet_idx])
print("Outflow values : ", mn[outlet_idx])
