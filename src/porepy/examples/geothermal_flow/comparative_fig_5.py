import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from porepy.examples.geothermal_flow.vtk_sampler import VTKSampler

figure_type = "vertical"

parametric_space_ref_level = 0
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

# --- 1. Load Data and Create Interpolators ---
print("\nStep 1: Loading data and creating interpolators...")

# Dictionary to hold data and interpolators
data_fields = {
    'Temperature': {'file': 'benchmark_figures_data/fig_5_'+figure_type+'_temperature_raw.csv', 'column': 'Temperature (°C)'},
    'Pressure': {'file': 'benchmark_figures_data/fig_5_'+figure_type+'_pressured_raw.csv', 'column': 'Pressure (MPa)'},
    'Saturation': {'file': 'benchmark_figures_data/fig_5_'+figure_type+'_saturation_liq_raw.csv', 'column': 'Liquid Saturation'}
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

# compute saturation
t_ref_v = df_interpolated['Temperature (°C)'].values + 273.15
s_ref_v = np.clip(df_interpolated['Liquid Saturation'].values, 0, 1)
p_v = df_interpolated['Pressure (MPa)'].values


def find_properties_by_saturation(s_ref_v, p_v, brine_sampler_phz, h_min=0.1, h_max=3.5, max_iterations=100, tolerance=1e-4):

    print("--- Starting Bisection Search for Enthalpy ---")
    num_points = len(s_ref_v)

    # Initialize lower (ha_v) and upper (hb_v) enthalpy bounds for all points
    ha_v = h_min * np.ones(num_points)
    hb_v = h_max * np.ones(num_points)

    for i in range(max_iterations):
        # Calculate the midpoint enthalpy for the current interval for all points
        h_mid_v = 0.5 * (ha_v + hb_v)

        # Prepare the input points for the model sampler
        par_points = np.array((np.zeros(num_points), h_mid_v, p_v)).T

        brine_sampler_phz.sample_at(par_points)
        s_calc_v = brine_sampler_phz.sampled_could.point_data["S_v"]
        s_diff_v = s_ref_v - s_calc_v

        # Check if all points have converged
        if np.all(np.abs(s_diff_v) < tolerance):
            print(f"Convergence reached for all points at iteration {i + 1}.")
            break
        ha_v = np.where(s_diff_v > 0.0, h_mid_v, ha_v)
        hb_v = np.where(s_diff_v <= 0.0, h_mid_v, hb_v)

        print(f"  Iteration {i + 1:2d}/{max_iterations}: Max |Δs| = {np.max(np.abs(s_diff_v)):.4f}")

    else:  # This 'else' belongs to the 'for' loop and runs if the loop completes without 'break'
        print("Warning: Maximum iterations reached without convergence for all points.")

    # The final converged enthalpy is the last midpoint
    h_final_v = 0.5 * (ha_v + hb_v)

    # Perform one final sampling with the converged enthalpy to get all final properties
    print("Performing final sampling with converged enthalpy values...")
    final_par_points = np.array((np.zeros(num_points), h_final_v, p_v)).T
    brine_sampler_phz.sample_at(final_par_points)

    t_final_v = brine_sampler_phz.sampled_could.point_data["Temperature"]
    s_gas_final_v = brine_sampler_phz.sampled_could.point_data["S_v"]

    print("--- Bisection Search Complete ---")

    return t_final_v, h_final_v, s_gas_final_v

def find_properties_by_temperature(t_ref_v, p_v, model, h_min=0.1, h_max=3.5, max_iterations=100, tolerance=1e-4):

    print("--- Starting Bisection Search for Enthalpy ---")
    num_points = len(t_ref_v)

    # Initialize lower (ha_v) and upper (hb_v) enthalpy bounds for all points
    ha_v = h_min * np.ones(num_points)
    hb_v = h_max * np.ones(num_points)

    for i in range(max_iterations):
        # Calculate the midpoint enthalpy for the current interval for all points
        h_mid_v = 0.5 * (ha_v + hb_v)

        # Prepare the input points for the model sampler
        par_points = np.array((np.zeros(num_points), h_mid_v, p_v)).T

        # Call the model to get the temperature based on the current enthalpy guess
        brine_sampler_phz.sample_at(par_points)
        t_calc_v = brine_sampler_phz.sampled_could.point_data["Temperature"]

        # Calculate the difference between reference and calculated saturation
        t_diff_v = t_ref_v - t_calc_v

        # Check if all points have converged
        if np.all(np.abs(t_diff_v) < tolerance):
            print(f"Convergence reached for all points at iteration {i + 1}.")
            break

        # Update the bounds for the next iteration using NumPy's `where` for efficiency.
        # If s_diff_v > 0, t_calc is too low, so we need higher enthalpy -> new lower bound is h_mid
        # If s_diff_v <= 0, t_calc is too high, so we need lower enthalpy -> new upper bound is h_mid
        ha_v = np.where(t_diff_v > 0.0, h_mid_v, ha_v)
        hb_v = np.where(t_diff_v <= 0.0, h_mid_v, hb_v)

        print(f"  Iteration {i + 1:2d}/{max_iterations}: Max |ΔT| = {np.max(np.abs(t_diff_v)):.4f} K")

    else:  # This 'else' belongs to the 'for' loop and runs if the loop completes without 'break'
        print("Warning: Maximum iterations reached without convergence for all points.")

    # The final converged enthalpy is the last midpoint
    h_final_v = 0.5 * (ha_v + hb_v)

    # Perform one final sampling with the converged enthalpy to get all final properties
    print("Performing final sampling with converged enthalpy values...")
    final_par_points = np.array((np.zeros(num_points), h_final_v, p_v)).T
    brine_sampler_phz.sample_at(final_par_points)

    t_final_v = brine_sampler_phz.sampled_could.point_data["Temperature"]
    s_gas_final_v = brine_sampler_phz.sampled_could.point_data["S_v"]

    print("--- Bisection Search Complete ---")

    return t_final_v, h_final_v, s_gas_final_v

mp_idx = np.where((1.0 - s_ref_v) * s_ref_v > 0.0)
sp_idx = np.where(np.isclose((1.0 - s_ref_v) * s_ref_v,0.0))

t_final_v, h_final_v, s_gas_final_v = np.zeros_like(p_v), np.zeros_like(p_v), np.zeros_like(p_v)

t_final_v[sp_idx], h_final_v[sp_idx], s_gas_final_v[sp_idx] = find_properties_by_temperature(t_ref_v[sp_idx], p_v[sp_idx], brine_sampler_phz)
t_final_v[mp_idx], h_final_v[mp_idx], s_gas_final_v[mp_idx] = find_properties_by_saturation(1.0-s_ref_v[mp_idx], p_v[mp_idx], brine_sampler_phz)
s_liq = 1.0 - s_gas_final_v

# --- 3. Plot the Interpolated Quantities ---
print("\nStep 3: Generating plots...")

# Create a figure with two subplots stacked vertically, sharing the x-axis
fig, (ax1, ax2) = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 10),
    sharex=True  # Both subplots will share the same x-axis
)
fig.suptitle('Thermodynamical quantities vs. Distance', fontsize=16)

# --- Subplot 1: Temperature and Pressure ---
ax1_twin = ax1.twinx()  # Create a secondary y-axis

# Plot Temperature on the primary y-axis (left)
color_temp = 'red'
ax1.set_ylabel('Temperature (°C)', color=color_temp)
ax1.plot(df_interpolated['Distance (km)'], t_final_v - 273.15, color=color_temp, marker='o',
         linestyle='-', label='Temperature')
ax1.tick_params(axis='y', labelcolor=color_temp)
ax1.grid(True, linestyle=':')

# Plot Pressure on the secondary y-axis (right)
color_pressure = 'blue'
ax1_twin.set_ylabel('Pressure (MPa)', color=color_pressure)
ax1_twin.plot(df_interpolated['Distance (km)'], df_interpolated['Pressure (MPa)'], color=color_pressure, marker='s',
              linestyle='--', label='Pressure')
ax1_twin.tick_params(axis='y', labelcolor=color_pressure)

ax2_twin = ax2.twinx()  # Create a secondary y-axis
# Plot Pressure on the secondary y-axis (right)
color_pressure = 'blue'
ax2_twin.set_ylabel('Enthalpy (MJ/Kg)', color=color_pressure)
ax2_twin.plot(df_interpolated['Distance (km)'], h_final_v, color=color_pressure, marker='s',
              linestyle='--', label='Enthalpy')
ax2_twin.tick_params(axis='y', labelcolor=color_pressure)

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
plt.savefig('reproduced_plot_'+figure_type+'.png')