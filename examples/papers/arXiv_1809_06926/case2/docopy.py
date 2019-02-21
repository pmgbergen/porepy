from shutil import copyfile
import os

if __name__ == "__main__":

    folder_src = "/Home/siv28/afu082/porepy/examples/papers/arXiv_1809_06926/case2/"
    folder_dst = folder_src + "/CSV/"

    solver_names = ["tpfa", "vem", "rt0", "mpfa"]
    refinements = ["0", "1", "2"]
    perms = ["0", "1"]

    for solver in solver_names:
        if solver == "vem":
            folder_out = folder_dst + "MVEM/"
        else:
            folder_out = folder_dst + solver.upper() + "/"

        for perm in perms:
            for refinement in refinements:
                folder_in = (
                    folder_src + solver + "_results_" + perm + "_" + refinement + "/"
                )

                if not os.path.exists(folder_out):
                    os.makedirs(folder_out)

                # copy a file
                name_src = "dol_perm_" + perm + "_refinement_" + refinement + ".csv"
                name_dst = name_src
                copyfile(folder_in + name_src, folder_out + name_dst)

            folder_in = folder_src + solver + "_results_" + perm + "_1/"

            # copy a file
            name_src = "dot_perm_" + perm + ".csv"
            name_dst = name_src
            copyfile(folder_in + name_src, folder_out + name_dst)

            # copy a file
            name_src = solver + "_results_perm_" + perm + ".csv"
            name_dst = "results_perm_" + perm + ".csv"
            copyfile(folder_src + name_src, folder_out + name_dst)
