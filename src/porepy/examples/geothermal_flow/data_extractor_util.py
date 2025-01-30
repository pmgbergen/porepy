import numpy as np
import os
import pyvista as pv
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def fig_4A_load_and_project_reference_data(xc):
    # doi: 10.1111/gfl.12080
    file_prefix = 'verification_pure_water/reference_data/'
    p_data = np.genfromtxt(file_prefix + 'fig_4a_pressure.csv', delimiter=',', skip_header=1)
    t_data = np.genfromtxt(file_prefix + 'fig_4a_temperature.csv', delimiter=',', skip_header=1)
    p_proj = np.interp(xc, p_data[:, 0], p_data[:, 1])
    t_proj = np.interp(xc, t_data[:, 0], t_data[:, 1])
    return p_proj, t_proj


def fig_4C_load_and_project_reference_data(xc):
    # doi: 10.1111/gfl.12080
    file_prefix = 'verification_pure_water/reference_data/'
    p_data = np.genfromtxt(file_prefix + 'fig_4c_pressure.csv', delimiter=',', skip_header=1)
    t_data = np.genfromtxt(file_prefix + 'fig_4c_temperature.csv', delimiter=',', skip_header=1)
    p_proj = np.interp(xc, p_data[:, 0], p_data[:, 1])
    t_proj = np.interp(xc, t_data[:, 0], t_data[:, 1])
    return p_proj, t_proj


def fig_4E_load_and_project_reference_data(xc):
    # doi: 10.1111/gfl.12080
    file_prefix = 'verification_pure_water/reference_data/'
    p_data = np.genfromtxt(file_prefix + 'fig_4e_pressure.csv', delimiter=',', skip_header=1)
    t_data = np.genfromtxt(file_prefix + 'fig_4e_temperature.csv', delimiter=',', skip_header=1)
    p_proj = np.interp(xc, p_data[:, 0], p_data[:, 1])
    t_proj = np.interp(xc, t_data[:, 0], t_data[:, 1])
    return p_proj, t_proj


def fig_5_load_and_project_reference_data(xc):
    # doi: 10.1111/gfl.12080
    file_prefix = 'verification_pure_water/reference_data/'
    p_data = np.genfromtxt(file_prefix + 'fig_5a_pressure.csv', delimiter=',', skip_header=1)
    t_data = np.genfromtxt(file_prefix + 'fig_5a_temperature.csv', delimiter=',', skip_header=1)
    sl_data = np.genfromtxt(file_prefix + 'fig_5b_liquid_saturation.csv', delimiter=',', skip_header=1)
    p_proj = np.interp(xc, p_data[:, 0], p_data[:, 1])
    t_proj = np.interp(xc, t_data[:, 0], t_data[:, 1])
    s_proj = np.interp(xc, sl_data[:, 0], sl_data[:, 1])
    return p_proj, t_proj, s_proj


def fig_6_load_and_project_reference_data(xc):
    # doi: 10.1111/gfl.12080
    file_prefix = 'verification_pure_water/reference_data/'
    p_data = np.genfromtxt(file_prefix + 'fig_6a_pressure.csv', delimiter=',', skip_header=1)
    t_data = np.genfromtxt(file_prefix + 'fig_6a_temperature.csv', delimiter=',', skip_header=1)
    sl_data = np.genfromtxt(file_prefix + 'fig_6b_liquid_saturation.csv', delimiter=',', skip_header=1)
    p_proj = np.interp(xc, p_data[:, 0], p_data[:, 1])
    t_proj = np.interp(xc, t_data[:, 0], t_data[:, 1])
    s_proj = np.interp(xc, sl_data[:, 0], sl_data[:, 1])
    return p_proj, t_proj, s_proj


def get_last_mesh_from_pvd(
    pvd_file: str
) -> pv.PolyData:
    """
    Reads a PVD file, extracts the last time step's VTU file, 
    and returns the corresponding PyVista mesh.

    Args:
        pvd_file (str): Path to the PVD file.

    Returns:
        pv.PolyData: PyVista mesh of the last time step.
    """

    # Ensure the PVD file exists
    if not os.path.exists(pvd_file):
        raise FileNotFoundError(f"PVD file not found: {pvd_file}")

    # Get directory of the PVD file
    pvd_dir = os.path.dirname(os.path.abspath(pvd_file))

    # Parse the PVD XML
    tree = ET.parse(pvd_file)
    root = tree.getroot()

    # Extract all VTU file references
    datasets = root.find("Collection").findall("DataSet")
    vtu_files = [ds.get("file") for ds in datasets]  # Extract file paths

    # Ensure at least one file exists
    if not vtu_files:
        raise ValueError("No VTU files found in the PVD file!")

    # Get the last time step's .vtu file
    last_vtu = vtu_files[-1]
    last_vtu_path = os.path.join(pvd_dir, last_vtu)  # Ensure correct absolute path

    # Check if the .vtu file exists
    if not os.path.exists(last_vtu_path):
        raise FileNotFoundError(f"Referenced VTU file not found: {last_vtu_path}")

    print(f"Last time step file: {last_vtu_path}")

    # Read the last .vtu file
    mesh = pv.read(last_vtu_path)
    
    return mesh


def plot_temp_pressure_comparison(
    mesh: pv.PolyData, 
    pressure_csmp : np.ndarray,
    temperature_csmp : np.ndarray,
    xc: np.array,
    y_limit_temp : list,
    temp_spacing : np.ndarray,
    y_limit_press: list,
    press_spacing : np.ndarray,
    simulation_time: float,
    save_path: str,
):
    """
    Plots temperature and pressure data from a PorePy simulation and compares it 
    to CSMP++ results.
    """

    # Plot settings
    linewidth_c = 3.5
    linewidth_p = 1.5
    fontsize = 23
    legend_cord = (1.01, 0.98)
    legend_fontsize = 13

    # Extract cell centers & scale x-coordinates to km
    centroids = mesh.cell_centers().points
    x_coords = centroids[:, 0] * 1e-3  # Convert to km

    # Extract 'pressure' and 'temperature' data (cell data)
    pressure = mesh.cell_data['pressure'] * 1e-6  # Convert to MPa
    temperature = mesh.cell_data['temperature'] - 273.15  # Convert to °C

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Plot Temperature (Left Y-Axis)
    color = 'tab:red'
    ax1.set_xlabel('Distance (km)', fontsize=fontsize)
    ax1.set_ylabel('Temperature (°C)', color=color, fontsize=fontsize)
    ax1.plot(x_coords, temperature, '-', color=color, label='Temperature (Porepy)', linewidth=linewidth_p)
    ax1.plot(x_coords, temperature_csmp, '--', color=color, label='Temperature (CSMP++)', linewidth=linewidth_c)
    ax1.tick_params(axis='y', labelcolor=color, direction='in')

    # Set x-axis and y-axis ticks
    ax1.set_xticks(np.linspace(0, 2, 5))
    ax1.tick_params(axis='x', direction='in')
    ax1.set_xticklabels(['0', '', '1', '', '2'], fontsize=14)
    ax1.set_yticks(temp_spacing)

    # Create Pressure Plot (Right Y-Axis)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Pressure (MPa)', color=color, fontsize=fontsize)
    ax2.plot(x_coords, pressure, '-', color=color, label='Pressure (Porepy)', linewidth=linewidth_p)
    ax2.plot(x_coords, pressure_csmp, '--', color=color, label='Pressure (CSMP++)', linewidth=linewidth_c)
    ax2.tick_params(axis='y', labelcolor=color, direction='in')
    ax2.set_yticks(press_spacing)

    # Add Legend
    fig.legend(loc="upper right", bbox_to_anchor=legend_cord, bbox_transform=ax1.transAxes, fontsize=legend_fontsize)

    # Set axis limits
    ax1.set_xlim([0, 2])
    ax1.set_ylim(y_limit_temp)
    ax2.set_ylim(y_limit_press)

    # Increase label font sizes
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    # Add a text annotation
    plt.text(0.7, 0.52, f"{simulation_time} + years", transform=ax1.transAxes, fontsize=fontsize)

    # Save and show the plot
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    print(f"Plot saved: {save_path}")


def plot_temp_pressure_two_phase(
    x_coords: np.ndarray,
    temperature: np.ndarray,
    temp_limit : np.ndarray,
    temp_spacing : np.ndarray,
    pressure: np.ndarray,
    press_limit : np.ndarray,
    press_spacing : np.ndarray,
    min_x: float,
    max_x: float,
    simulation_time : float,
    simulator: str,
    save_path: str
):
    """
    Plots temperature, pressure, and the two-phase region from PorePy simulation results.
    """

    # Plot settings
    linewidth_c = 3.5
    linewidth_p = 3.5
    fontsize = 23
    legend_cord = (1.09, 0.98)
    legend_fontsize = 13

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Determine line style based on dataset
    line_style = '-' if simulator.lower() == "porepy" else '--'

    # Plot Temperature (Left Y-Axis)
    color = 'tab:red'
    ax1.set_xlabel('Distance (km)', fontsize=fontsize)
    ax1.set_ylabel('Temperature (°C)', color=color, fontsize=fontsize)
    ax1.plot(x_coords, temperature, line_style, color=color, label=f'Temperature ({simulator})', linewidth=linewidth_c)
    ax1.tick_params(axis='y', labelcolor=color, direction='in')

    # Set x-axis and y-axis ticks
    ax1.set_xticks(np.linspace(0, 2, 5))
    ax1.tick_params(axis='x', direction='in')
    ax1.set_xticklabels(['0', '', '1', '', '2'])
    ax1.set_yticks(temp_spacing)

    # Create Pressure Plot (Right Y-Axis)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Pressure (MPa)', color=color, fontsize=fontsize)
    ax2.plot(x_coords, pressure, line_style, color=color, label=f'Pressure ({simulator})', linewidth=linewidth_p)
    ax2.tick_params(axis='y', labelcolor=color, direction='in')
    ax2.set_yticks(press_spacing)

    # Add Grid & Set Axis Limits
    ax1.grid(False)
    ax1.set_xlim([0, 2])  # X-axis limit
    ax1.set_ylim(temp_limit)  # Temperature Y-axis limit
    ax2.set_ylim(press_limit)  # Pressure Y-axis limit

    ### TWO-PHASE REGION
    # Shade the two-phase region
    ax1.axvspan(min_x, max_x, color='gray', alpha=0.5)

    # Add annotation (optional)
    plt.text(0.6, 0.7, f"{simulation_time} + years", transform=ax1.transAxes, fontsize=fontsize)

    # Add Legend
    fig.legend(loc="upper right", bbox_to_anchor=legend_cord, bbox_transform=ax1.transAxes, fontsize=legend_fontsize)

    # Increase label font sizes
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    print(f"Plot saved at: {save_path}")


def plot_liquid_saturation(
    x_coords: np.ndarray,
    s_liq: np.ndarray,
    min_x: float,
    max_x: float,
    save_path: str,
    simulator: str = "porepy",  # "porepy" -> solid line, "csmp" -> dashed line
):
    """
    Plots liquid saturation for either CSMP++ or PorePy and saves the plot.
    """

    # Plot settings
    linewidth_p = 3.5
    fontsize = 23
    legend_cord = (1.1, 0.9)
    legend_fontsize = 13

    # Determine line style based on dataset
    line_style = '-' if simulator.lower() == "porepy" else '--'

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # Plot Liquid Saturation
    color = 'tab:green'
    ax1.set_xlabel('Distance (km)', fontsize=fontsize)
    ax1.set_ylabel('Liquid Saturation', color=color, fontsize=fontsize)
    ax1.plot(x_coords, s_liq, line_style, color=color, label=f'Liquid Saturation ({simulator})', linewidth=linewidth_p)
    ax1.tick_params(axis='y', labelcolor=color, direction='in')

    # X-Axis Formatting
    ax1.set_xticks(np.linspace(0, 2, 5))
    ax1.tick_params(axis='x', direction='in')
    ax1.set_xticklabels(['0', '', '1', '', '2'])

    # Y-Axis Formatting
    ax1.set_yticks(np.linspace(0, 1, 5))
    ax1.set_xlim([0, 2])  # X-axis limit
    ax1.set_ylim([-0.2e-1, 1.01])  # Liquid saturation Y-axis limit

    # Hide Grid
    ax1.grid(False)

    ### **TWO-PHASE REGION**
    # Shade the region where min_x <= x <= max_x
    ax1.axvspan(min_x, max_x, color='gray', alpha=0.5)

    # Add Residual Liquid Saturation Line
    ax1.axhline(y=0.3, color='black', linestyle='--', linewidth=1.8)

    ### **Annotations**
    color = 'black'
    plt.text(0.055, 0.6, 'Vapor', transform=ax1.transAxes, fontsize=fontsize, verticalalignment='center', color=color)
    plt.text(0.75, 0.6, 'Liquid', transform=ax1.transAxes, fontsize=fontsize, verticalalignment='center', color=color)
    plt.text(0.45, 0.6, 'Vapor + Liquid', transform=ax1.transAxes, rotation=90, fontsize=14, verticalalignment='center', color=color)
    plt.text(1.3, 0.32, 'Residual liquid', fontsize=13, color='black', verticalalignment='center')
    plt.text(1.36, 0.28, 'saturation', fontsize=13, color='black', verticalalignment='center')

    # Add Legend
    fig.legend(loc="upper right", bbox_to_anchor=legend_cord, bbox_transform=ax1.transAxes, fontsize=legend_fontsize)

    # Increase label font sizes
    ax1.tick_params(axis='both', which='major', labelsize=14)

    # Save the plot
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
    print(f"Plot saved at: {save_path}")

