from shutil import copyfile

folder_src = "/home/elle/Dropbox/Work/PresentazioniArticoli/2019/Articles/tipetut++/Results/example1/img/"
folder_dist = "/home/elle/Dropbox/Work/PresentazioniArticoli/2019/Articles/tipetut++/Article/Parts/Images/"

# the number of cells
grid = [0, 1, 2]
grids_label = {0: "coarse", 1: "medium", 2: "fine"}

for g in grid:
    name_src = grids_label[g] + "_num_cells.png"
    name_dist = "example1_" + name_src
    copyfile(folder_src + name_src, folder_dist + name_dist)

# outflow
grid = [0, 1, 2]
config = [0, 10, 20]

for g in grid:
    for c in config:
        name_src = grids_label[g] + "_outflow_" + str(c) + ".png"
        name_dist = "example1_" + name_src
        copyfile(folder_src + name_src, folder_dist + name_dist)

# avgerage temperature
grid = [0, 1, 2]
config = [0, 10, 20]
frac = [0]

for g in grid:
    for c in config:
        for f in frac:
            name_src = (
                grids_label[g] + "_cot_avg_" + str(c) + "_frac_" + str(f) + ".png"
            )
            name_dist = "example1_" + name_src
            copyfile(folder_src + name_src, folder_dist + name_dist)

# minimum temperature
grid = [0, 1, 2]
config = [0, 10, 20]
frac = [0]

for g in grid:
    for c in config:
        for f in frac:
            name_src = (
                grids_label[g] + "_cot_min_" + str(c) + "_frac_" + str(f) + ".png"
            )
            name_dist = "example1_" + name_src
            copyfile(folder_src + name_src, folder_dist + name_dist)

# maximum temperature
grid = [0, 1, 2]
config = [0, 10, 20]
frac = [0]

for g in grid:
    for c in config:
        for f in frac:
            name_src = (
                grids_label[g] + "_cot_max_" + str(c) + "_frac_" + str(f) + ".png"
            )
            name_dist = "example1_" + name_src
            copyfile(folder_src + name_src, folder_dist + name_dist)
