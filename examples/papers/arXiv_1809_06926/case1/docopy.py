from shutil import copyfile
import os

if __name__ == "__main__":

    folder_src = "/Home/siv28/afu082/porepy/examples/papers/arXiv_1809_06926/case1/"
    folder_dst = folder_src + "/CSV/"

    solver_names = ["tpfa", "vem", "rt0", "mpfa"]
    refinements = ["0", "1", "2"]

    for refinement in refinements:
        for solver in solver_names:
            folder_in = folder_src + solver + "_results_" + refinement + "/"
            if solver == "vem":
                folder_out = folder_dst + "MVEM/"
            else:
                folder_out = folder_dst + solver.upper() + "/"

            if not os.path.exists(folder_out):
                os.makedirs(folder_out)

            # copy a file
            name_src = "dol_refinement_" + refinement + ".csv"
            name_dst = name_src
            copyfile(folder_in + name_src, folder_out + name_dst)

            # copy a file
            name_src = "dot_refinement_" + refinement + ".csv"
            name_dst = name_src
            copyfile(folder_in + name_src, folder_out + name_dst)

            # copy a file
            name_src = solver + "_results.csv"
            name_dst = "results.csv"
            copyfile(folder_src + name_src, folder_out + name_dst)
