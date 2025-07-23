import pandas as pd
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

# --- Configuration ---
figure_type = "horizontal"


def extract_cell_data(file_path: str, field_names: list) -> dict:
    """
    Reads a VTU file and extracts specified data fields from the cell data.

    Args:
        file_path (str): The full path to the .vtu file.
        field_names (list): A list of strings with the names of the cell data
                            fields to extract.

    Returns:
        dict: A dictionary containing the cell center coordinates (key 'xc')
              scaled to kilometers, and the requested data fields. Returns an
              empty dictionary if the file is not found.
    """
    try:
        mesh = pv.read(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return {}

    # Get the coordinates of the cell centers and scale from m to km
    cell_centers = mesh.cell_centers().points
    extracted_data = {'xc': cell_centers / 1000.0}

    # Extract each requested field from the cell data
    for field in field_names:
        if field in mesh.cell_data:
            extracted_data[field] = mesh.cell_data[field]
        else:
            print(
                f"Warning: Field '{field}' not found. Available fields: {list(mesh.cell_data.keys())}")

    return extracted_data


def find_properties_by_saturation(s_ref_v, p_v, sampler, h_min=0.1, h_max=3.5, max_iterations=100, tolerance=1e-4):
    """
    Finds thermodynamic properties by performing a bisection search for enthalpy
    that matches a reference gas saturation.

    Args:
        s_ref_v (np.ndarray): Array of reference gas saturation values.
        p_v (np.ndarray): Array of pressure values (MPa).
        sampler (VTKSampler): An initialized VTKSampler object for phz space.
        h_min (float): Minimum bound for enthalpy search (MJ/kg).
        h_max (float): Maximum bound for enthalpy search (MJ/kg).
        max_iterations (int): Maximum number of bisection iterations.
        tolerance (float): Convergence tolerance for saturation difference.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
        final temperature (K), enthalpy (MJ/kg), and gas saturation arrays.
    """
    print("--- Starting Bisection Search for Enthalpy (matching Saturation) ---")
    num_points = len(s_ref_v)
    ha_v = h_min * np.ones(num_points)
    hb_v = h_max * np.ones(num_points)

    for i in range(max_iterations):
        h_mid_v = 0.5 * (ha_v + hb_v)
        par_points = np.array((np.zeros(num_points), h_mid_v, p_v)).T
        sampler.sample_at(par_points)
        s_calc_v = sampler.sampled_could.point_data["S_v"]
        s_diff_v = s_ref_v - s_calc_v

        if np.all(np.abs(s_diff_v) < tolerance):
            print(f"Convergence reached for all points at iteration {i + 1}.")
            break

        ha_v = np.where(s_diff_v > 0.0, h_mid_v, ha_v)
        hb_v = np.where(s_diff_v <= 0.0, h_mid_v, hb_v)
        print(f"  Iteration {i + 1:2d}/{max_iterations}: Max |Δs| = {np.max(np.abs(s_diff_v)):.4f}")
    else:
        print("Warning: Maximum iterations reached without full convergence.")

    h_final_v = 0.5 * (ha_v + hb_v)
    final_par_points = np.array((np.zeros(num_points), h_final_v, p_v)).T
    sampler.sample_at(final_par_points)
    t_final_v = sampler.sampled_could.point_data["Temperature"]
    s_gas_final_v = sampler.sampled_could.point_data["S_v"]
    print("--- Bisection Search Complete ---")
    return t_final_v, h_final_v, s_gas_final_v


def find_properties_by_temperature(t_ref_v, p_v, sampler, h_min=0.1, h_max=3.5, max_iterations=100, tolerance=1e-4):
    """
    Finds thermodynamic properties by performing a bisection search for enthalpy
    that matches a reference temperature.

    Args:
        t_ref_v (np.ndarray): Array of reference temperature values (K).
        p_v (np.ndarray): Array of pressure values (MPa).
        sampler (VTKSampler): An initialized VTKSampler object for phz space.
        h_min (float): Minimum bound for enthalpy search (MJ/kg).
        h_max (float): Maximum bound for enthalpy search (MJ/kg).
        max_iterations (int): Maximum number of bisection iterations.
        tolerance (float): Convergence tolerance for temperature difference (K).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
        final temperature (K), enthalpy (MJ/kg), and gas saturation arrays.
    """
    print("--- Starting Bisection Search for Enthalpy (matching Temperature) ---")
    num_points = len(t_ref_v)
    ha_v = h_min * np.ones(num_points)
    hb_v = h_max * np.ones(num_points)

    for i in range(max_iterations):
        h_mid_v = 0.5 * (ha_v + hb_v)
        par_points = np.array((np.zeros(num_points), h_mid_v, p_v)).T
        sampler.sample_at(par_points)
        t_calc_v = sampler.sampled_could.point_data["Temperature"]
        t_diff_v = t_ref_v - t_calc_v

        if np.all(np.abs(t_diff_v) < tolerance):
            print(f"Convergence reached for all points at iteration {i + 1}.")
            break

        ha_v = np.where(t_diff_v > 0.0, h_mid_v, ha_v)
        hb_v = np.where(t_diff_v <= 0.0, h_mid_v, hb_v)
        print(f"  Iteration {i + 1:2d}/{max_iterations}: Max |ΔT| = {np.max(np.abs(t_diff_v)):.4f} K")
    else:
        print("Warning: Maximum iterations reached without full convergence.")

    h_final_v = 0.5 * (ha_v + hb_v)
    final_par_points = np.array((np.zeros(num_points), h_final_v, p_v)).T
    sampler.sample_at(final_par_points)
    t_final_v = sampler.sampled_could.point_data["Temperature"]
    s_gas_final_v = sampler.sampled_could.point_data["S_v"]
    print("--- Bisection Search Complete ---")
    return t_final_v, h_final_v, s_gas_final_v


# --- 1. Load Numerical and Reference Data ---
print("\nStep 1: Loading data...")

# 1a. Load data from PorePy's VTU output file

vtk_map = {'horizontal': 'fig5_horizontal_time_idx_000200_l0', 'vertical': 'fig5_vertical_time_idx_004000_l0'}

vtk_file = "benchmark_figures_data/porepy_vtks/" + vtk_map[figure_type] + ".vtu"
fields_to_extract = ['pressure', 'temperature', 's_gas', 'enthalpy']
pp_data = extract_cell_data(vtk_file, fields_to_extract)
idx_map = {'horizontal': 0, 'vertical': 1}
distance_xc = pp_data['xc'][:, idx_map[figure_type]]

# 1b. Load reference data from benchmark CSV files
data_fields = {
    'Temperature': {'file': f'benchmark_figures_data/fig_5_{figure_type}_temperature_raw.csv', 'column': 'Temperature (°C)'},
    'Pressure': {'file': f'benchmark_figures_data/fig_5_{figure_type}_pressured_raw.csv', 'column': 'Pressure (MPa)'},
    'Saturation': {'file': f'benchmark_figures_data/fig_5_{figure_type}_saturation_liq_raw.csv', 'column': 'Liquid Saturation'}
}

# 1c. Create interpolators for the reference data
interpolators = {}
for key, field in data_fields.items():
    try:
        df_raw = pd.read_csv(field['file'])
        x_raw = df_raw['Distance (km)']
        y_raw = df_raw[field['column']]
        interpolators[key] = interp1d(x_raw, y_raw, kind='linear', fill_value="extrapolate")
        print(f"  - Interpolator for '{key}' created.")
    except FileNotFoundError:
        print(f"Error: The file '{field['file']}' was not found.")
        exit()

# --- 2. Resample Data and Compute Thermodynamic Properties ---
print("\nStep 2: Resampling data and computing properties...")

# 2a. Define a regular grid and resample reference data onto it
resolution = 0.025
distance_regular = np.arange(0.0, 2.0 + resolution, resolution)

df_interpolated = pd.DataFrame({'Distance (km)': distance_regular})
df_interpolated['Temperature (°C)'] = interpolators['Temperature'](distance_regular)
df_interpolated['Pressure (MPa)'] = interpolators['Pressure'](distance_regular)
df_interpolated['Liquid Saturation'] = interpolators['Saturation'](distance_regular)

# 2b. Prepare inputs for thermodynamic calculations
t_ref_v = df_interpolated['Temperature (°C)'].values + 273.15  # Convert to Kelvin
s_liq_ref_v = np.clip(df_interpolated['Liquid Saturation'].values, 0, 1)
p_v = df_interpolated['Pressure (MPa)'].values
s_gas_ref_v = 1.0 - s_liq_ref_v

# 2c. Set up the brine property sampler
parametric_space_ref_level = 0
file_name_prefix = "model_configuration/constitutive_description/driesner_vtk_files/"
file_name_phz = f"{file_name_prefix}XHP_l{parametric_space_ref_level}_simplex_iapws.vtk"
brine_sampler_phz = VTKSampler(file_name_phz)
brine_sampler_phz.conversion_factors = (1.0, 1.0, 1.0)  # (z,h,p)

# 2d. Identify single-phase and two-phase regions
# Tolerance to handle floating point inaccuracies near 0 and 1
is_two_phase = (s_liq_ref_v > 1e-6) & (s_liq_ref_v < 1 - 1e-6)
mp_idx = np.where(is_two_phase)
sp_idx = np.where(~is_two_phase)

# 2e. Calculate final properties based on phase
t_final_v, h_final_v, s_gas_final_v = np.zeros_like(p_v), np.zeros_like(p_v), np.zeros_like(p_v)

# For single-phase points, find enthalpy that matches the reference temperature
if sp_idx[0].size > 0:
    t_final_v[sp_idx], h_final_v[sp_idx], s_gas_final_v[sp_idx] = find_properties_by_temperature(
        t_ref_v[sp_idx], p_v[sp_idx], brine_sampler_phz
    )
# For two-phase points, find enthalpy that matches the reference saturation
if mp_idx[0].size > 0:
    t_final_v[mp_idx], h_final_v[mp_idx], s_gas_final_v[mp_idx] = find_properties_by_saturation(
        s_gas_ref_v[mp_idx], p_v[mp_idx], brine_sampler_phz
    )

# --- 3. Plot the Results ---
print("\nStep 3: Generating plots...")

# Create a figure with two square subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(8, 14),  # Adjusted for two square plots + text
    sharex=True
)
fig.suptitle(f'{figure_type.capitalize()}', fontsize=16)

# --- Subplot 1: Temperature and Pressure ---
ax1.set_title("Temperature and Pressure")
# FIX: Create the twin axes BEFORE setting the aspect ratio
ax1_twin = ax1.twinx()
ax1.set_box_aspect(1.0)

color_temp = 'red'
ax1.set_ylabel('Temperature (°C)', color=color_temp)
ax1.plot(distance_regular, t_final_v - 273.15, color=color_temp, linestyle='-', label='Temperature (Ref)')
ax1.plot(distance_xc, pp_data['temperature'] - 273.15, color=color_temp, linestyle='--', label='Temperature (Num)')
ax1.tick_params(axis='y', labelcolor=color_temp)
ax1.grid(True, linestyle=':')

color_pressure = 'blue'
ax1_twin.set_ylabel('Pressure (MPa)', color=color_pressure)
ax1_twin.plot(distance_regular, p_v, color=color_pressure, linestyle='-', label='Pressure (Ref)')
ax1_twin.plot(distance_xc, pp_data['pressure'], color=color_pressure, linestyle='--', label='Pressure (Num)')
ax1_twin.tick_params(axis='y', labelcolor=color_pressure)

# --- Subplot 2: Liquid Saturation and Enthalpy ---
ax2.set_title("Saturation and Enthalpy")
# FIX: Create the twin axes BEFORE setting the aspect ratio
ax2_twin = ax2.twinx()
ax2.set_box_aspect(1.0)

color_sat = 'green'
ax2.set_ylabel('Liquid Saturation', color=color_sat)
ax2.plot(distance_regular, 1.0 - s_gas_final_v, color=color_sat, linestyle='-', label='Liquid Saturation (Ref)')
ax2.plot(distance_xc, 1.0 - pp_data['s_gas'], color=color_sat, linestyle='--', label='Liquid Saturation (Num)')
ax2.tick_params(axis='y', labelcolor=color_sat)
ax2.set_ylim(0, 1.1)
ax2.grid(True, linestyle=':')

color_enthalpy = 'purple'
ax2_twin.set_ylabel('Enthalpy (MJ/kg)', color=color_enthalpy)
ax2_twin.plot(distance_regular, h_final_v, color=color_enthalpy, linestyle='-', label='Enthalpy (Ref)')
ax2_twin.plot(distance_xc, pp_data['enthalpy'], color=color_enthalpy, linestyle='--', label='Enthalpy (Num)')
ax2_twin.tick_params(axis='y', labelcolor=color_enthalpy)

# --- Final Touches ---
ax2.set_xlabel('Distance (km)', fontsize=12)

# Create unified legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines1_twin, labels1_twin = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines1_twin, labels1 + labels1_twin, loc='center right')

lines2, labels2 = ax2.get_legend_handles_labels()
lines2_twin, labels2_twin = ax2_twin.get_legend_handles_labels()
ax2.legend(lines2 + lines2_twin, labels2 + labels2_twin, loc='center right')

fig.tight_layout(rect=[0, 0, 1, 0.96])
output_filename = f'reproduced_figure_5_{figure_type}.png'
plt.savefig(output_filename)
print(f"\nPlot saved as '{output_filename}'")
# plt.show()