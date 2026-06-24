#!/bin/bash
NUM = "17"
# Isothermal models.
echo "--- SIMULATION 0 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2a.py -a 1.1
echo "--- SIMULATION 1 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2a.py -a 1.5
echo "--- SIMULATION 2 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2a.py -a 2
echo "--- SIMULATION 3 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2a.py -a 5
echo "--- SIMULATION 4 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2a.py -a 10
echo "--- SIMULATION 5 / ${NUM} COMPLETED ---"
# Expecting failure: extensives eliminated and no preconditioning.
python3 src/porepy/examples/cold_injection/run_case2a.py -e -a 10
echo "--- SIMULATION 6 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2a.py -p -a 10
echo "--- SIMULATION 7 / ${NUM} COMPLETED ---"
# Thermal UV cases
# Thermal hybrid cases
# Plotting and printing results
python3 src/porepy/examples/cold_injection/plot_case2.py
echo "--- RUN CASE 2 COMPLETED ---"