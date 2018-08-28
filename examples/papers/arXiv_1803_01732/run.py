import os

methods = ["vem", "vem_coarse", "mpfa", "tpfa"]
examples = ["example_2_1_", "example_2_2_"]

run_log = "run_log.txt"
os.system("rm -rf " + run_log)

with open(run_log, "a") as f:
    for example in examples:
        for method in methods:
            cmd = "python -O " + example + method + ".py"
            f.write(cmd + "\n")
            os.system(cmd)
