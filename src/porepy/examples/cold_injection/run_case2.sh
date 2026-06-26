#!/bin/bash
NUM=15
# Isothermal models.
# pT based models
echo "--- SIMULATION 0 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2a.py -a 3
# Expecting failure: vT npc resolves gas, but global model failes to converge.
echo "--- SIMULATION 1 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2a.py -p -a 3
# vT based models
echo "--- SIMULATION 2 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2b.py -a 1.1
echo "--- SIMULATION 3 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2b.py -a 1.5
echo "--- SIMULATION 4 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2b.py -a 2
echo "--- SIMULATION 5 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2b.py -a 2.5
echo "--- SIMULATION 6 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2b.py -a 3
echo "--- SIMULATION 7 / ${NUM} COMPLETED ---"
# Expecting failure: extensives eliminated
python3 src/porepy/examples/cold_injection/run_case2b.py -e -a 3
echo "--- SIMULATION 8 / ${NUM} COMPLETED ---"
# Expecting success but no transient resolution: no nonlinear preconditioning
python3 src/porepy/examples/cold_injection/run_case2b.py -p -a 3
echo "--- SIMULATION 9 / ${NUM} COMPLETED ---"
# # Thermal UV cases
# python3 src/porepy/examples/cold_injection/run_case2c.py -a 1.1
# echo "--- SIMULATION 10 / ${NUM} COMPLETED ---"
# python3 src/porepy/examples/cold_injection/run_case2c.py -a 1.5
# echo "--- SIMULATION 11 / ${NUM} COMPLETED ---"
# python3 src/porepy/examples/cold_injection/run_case2c.py -a 2
# echo "--- SIMULATION 12 / ${NUM} COMPLETED ---"
# python3 src/porepy/examples/cold_injection/run_case2c.py -a 2.5
# echo "--- SIMULATION 13 / ${NUM} COMPLETED ---"
# python3 src/porepy/examples/cold_injection/run_case2c.py -a 3
# echo "--- SIMULATION 14 / ${NUM} COMPLETED ---"
# # Thermal hybrid cases: global ph formulation with uv pre-conditioning
# python3 src/porepy/examples/cold_injection/run_case2d.py -a 3
# echo "--- SIMULATION 15 / ${NUM} COMPLETED ---"
# Plotting and printing results
python3 src/porepy/examples/cold_injection/plot_case2.py
echo "--- RUN CASE 2 COMPLETED ---"