from shutil import copyfile
import os

if __name__ == "__main__":

    folder_src = "/Home/siv28/afu082/porepy/examples/papers/arXiv_1809_06926/case4/"
    folder_dst = folder_src + "/CSV/"

    solver_names = ["tpfa", "vem", "rt0", "mpfa"]

    for solver in solver_names:
        folder_in = folder_src + solver + "_results/"
        if solver == "vem":
            folder_out = folder_dst + "MVEM/"
        else:
            folder_out = folder_dst + solver.upper() + "/"

        if not os.path.exists(folder_out):
            os.makedirs(folder_out)

        # copy a file
        name_src = "dol_line_0.csv"
        name_dst = name_src
        copyfile(folder_in + name_src, folder_out + name_dst)

        # copy a file
        name_src = "dol_line_1.csv"
        name_dst = name_src
        copyfile(folder_in + name_src, folder_out + name_dst)

        # copy a file
        name_src = "dot.csv"
        name_dst = name_src
        copyfile(folder_in + name_src, folder_out + name_dst)

        # copy a file
        name_src = "results.csv"
        name_dst = name_src
        copyfile(folder_in + name_src, folder_out + name_dst)
