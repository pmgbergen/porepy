#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:00:38 2018

@author: eke001
"""

import paraview.simple as pv
import os, glob

import vtk
from vtk.util.numpy_support import vtk_to_numpy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_file(file_in):
    vtk_reader = vtk.vtkXMLUnstructuredGridReader()
    vtk_reader.SetFileName(file_in)
    vtk_reader.Update()
    return vtk_reader

#------------------------------------------------------------------------------#

def read_data(file_name, field):
    vtk_reader = read_file(file_name)
    data = vtk_reader.GetOutput().GetCellData().GetArray(field)
    return vtk_to_numpy(data)



def plot_over_line(file_in, file_out, pts, resolution=5000, fields=None):

    if file_in.lower().endswith('.pvd'):
        # create a new 'PVD Reader'
        sol = pv.PVDReader(FileName=file_in)
    elif file_in.lower().endswith('.vtu'):
        # create a new 'XML Unstructured Grid Reader'
        sol = pv.XMLUnstructuredGridReader(FileName=file_in)
    else:
        raise ValueError("file format not yet supported")

    # create a new 'Plot Over Line'
    pol = pv.PlotOverLine(Input=sol, Source='High Resolution Line Source')

    # Properties modified on plotOverLine1.Source
    pol.Source.Point1 = pts[0]
    pol.Source.Point2 = pts[1]
    pol.Source.Resolution = resolution

    # save data
    pv.SaveData(file_out, proxy=pol, Precision=15)
    if fields is not None:
        if not isinstance(fields, list):
            fields = list(fields)
        data = pd.read_csv(file_out)
        data = data[fields]
        data.to_csv(file_out)

#------------------------------------------------------------------------------#


def process_vtu_files(num_time_steps, plot_line=False):
    solver_names = ['tpfa', 'mpfa']#, 'vem', 'rt0']
    meshes = ['0', '1', '2', '3', '4']
    time_steps = ['0.1', '0.05', '0.025', '0.01', '0.005', '0.001']

    directories = []

    for item in os.listdir():
        if os.path.isdir(item):
            split = item.split('_')
            if len(split) < 4:
                continue
            if split[0] in solver_names and split[2] in meshes and split[4] in time_steps:
                directories.append(item)

    #directories = directories[:3]
    # Loop over all directories
    for run in directories:
        os.chdir(run)

        if plot_line:
            ### Generate plots over line
            diagonal_points = [[0, 0, 0], [1, 2.25, 1]]
            plot_over_line('sol.pvd', 'plot_over_line_diagonal.csv', diagonal_points, fields=['pressure'])

        #### Generate data for time evolution of mean tracer concentration in fractures

        # Read cell volumes
        volumes_2d = read_data("sol_2.vtu", 'cell_volumes')

        # Which fracture does a 2D cell belong to
        fracture_of_cell = read_data("sol_2.vtu", 'grid_node_number') - 1
        num_fracs = np.unique(fracture_of_cell).size

        # Brutal way to get the number of time steps
        max_time = -1
        for item in os.listdir():
            split = item.split('_')
            # Only consider tracer data
            if split[0] != 'tracer':
                continue
            # But not on the mortar grids
            if split[1] == 'mortars':
                continue
            time = int(split[2].split('.')[0])
            max_time = np.maximum(time, max_time)

        fracture_concentrations = np.zeros((num_fracs, num_time_steps))
        all_times = np.zeros(num_time_steps)
        time_counter = 0
        for item in os.listdir():

            split = item.split('_')
            # Only consider tracer data
            if split[0] != 'tracer':
                continue
            # But not on the mortar grids
            if split[1] == 'mortar':
                continue
            dim = int(split[1])
            if dim != 2:
                continue
            time = int(split[2].split('.')[0])
            concentrations = read_data(item, 'tracer')
            all_times[time_counter] = time
            for fi in range(num_fracs):
                hit = fracture_of_cell == fi
                mean_conc = np.sum(concentrations[hit] * volumes_2d[hit]) / np.sum(volumes_2d[hit])
                fracture_concentrations[fi, time_counter] = mean_conc
            time_counter += 1

        sort_ind = np.argsort(all_times)
        fracture_concentrations = fracture_concentrations[:, sort_ind]
        np.savetxt('mean_trace_fractures.csv', fracture_concentrations)


        # Done with this directory, move back again
        os.chdir('..')
    return directories



if __name__ == '__main__':

    t = np.arange(11) * 0.1
    num_time_steps = t.size


    plot_time_conv = False
    plot_space_conv = False
    compare_outflows = True

    if plot_time_conv or plot_space_conv:
        directories = process_vtu_files(num_time_steps)

    sns.set()


    do_save = True

    # Fracture concentrations for all simulations
    frac_conc = {}
    num_frac = -1
    all_meshes = []
    all_methods = []
    all_timesteps = []
    for item in directories:
        os.chdir(item)

        split = item.split('_')

        method = split[0]
        all_methods.append(method)
        mesh_ind = split[2]
        all_meshes.append(mesh_ind)
        time_step = split[4]
        all_timesteps.append(time_step)
        conc = np.genfromtxt('mean_trace_fractures.csv')

        plot_time = np.arange(conc.shape[1]) * float(time_step)
        num_frac = conc.shape[0]

        frac_conc[item] = {'time_steps': plot_time, 'values': conc}

        os.chdir('..')



    fracs_to_plot = np.array([2, 3, 4, 7])
    num_frac_plot = fracs_to_plot.size

    all_methods = set(all_methods)
    all_timesteps = set(all_timesteps)
    finest_timestep = np.min(np.array([float(i) for i in all_timesteps]))
    # Check convergence in time
    if plot_time_conv:
        for met in set(all_methods):
            for mesh in set(all_meshes):
                fig_arr = np.array([plt.figure() for i in fracs_to_plot])
                legend = np.array([{} for i in fracs_to_plot])
                do_plot = False
                for key, val in frac_conc.items():
                    split = key.split('_')
                    if split[0] != met or split[2] != mesh:
                        continue
                    do_plot = True
                    time_step = split[4]
                    for fi in range(num_frac_plot):
                        plt.figure(fig_arr[fi].number)
                        v, = plt.plot(t, val['values'][fi], label=time_step)
                        legend[fi][time_step] = v

                for fi in range(num_frac_plot):
                    plt.figure(fig_arr[fi].number)
                    if do_plot:
                        plt.title(met + ' ' + mesh + 'fracture ' + str(fracs_to_plot[fi]))
                        plt.legend(list(legend[fi].values()), list(legend[fi].keys()))
                        #plt.legend(handles=list(legend[fi].values()))
                        plt.show()


                    else:
                        plt.close()

    num_cells = np.array(['36000', '47274', '68147', '109430', '152247'])
    if plot_space_conv:

        for met in set(all_methods):
            fig_arr = np.array([plt.figure() for i in fracs_to_plot])
            legend = np.array([{} for i in fracs_to_plot])
            do_plot = False
            for key, val in frac_conc.items():
                split = key.split('_')
                if split[0] != met or float(split[4]) > finest_timestep:
                    continue
                do_plot = True
                mesh_type = split[2]
                for fi in range(num_frac_plot):
                    plt.figure(fig_arr[fi].number)
                    v, = plt.plot(t, val['values'][fi], label=mesh_type)
                    legend[fi][mesh_type] = num_cells[int(mesh_type)]

            for fi in range(num_frac_plot):
                plt.figure(fig_arr[fi].number)
                if do_plot:
                    plt.title(met + ' fracture ' + str(fracs_to_plot[fi]))
                    plt.legend(list(legend[fi].values()))
                    #plt.legend(handles=list(legend[fi].values()))
                    plt.show()
                    if do_save:
                        path = 'figures/Spatial_convergence_' + met + '_facture_' + str(fracs_to_plot[fi]) + '.png'
                        plt.savefig(path)
                else:
                    plt.close()

    if compare_outflows:

        vals = {m: np.empty((0, 3)) for m in all_methods}
        all_cells = []
        for name in glob.glob('concentrations*.txt'):

            split = name.replace(' ', '_').split('_')
            method = split[1]
            cells = int(split[2])
            all_cells.append(cells)
            time_step = split[4][:-4]
            if float(time_step) > finest_timestep:
                continue
            else:
                c = np.genfromtxt(name, delimiter=',')
                vals[method] = np.vstack((vals[method], np.hstack((cells, c))))

        for m, d in vals.items():
            sort_ind = np.argsort(d[:, 0])
            df = pd.DataFrame(d[sort_ind])
            df.columns = ['cells', 'lower', 'upper']
            df.to_csv('outflows_' + method + '.csv', index=False)