#!/bin/bash
NUM=23
# isothermal pT-model
echo "--- SIMULATION 0 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2a.py -n -j 3
echo "--- SIMULATION 1 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2a.py -j 3
echo "--- SIMULATION 2 / ${NUM} COMPLETED ---"
# isothermal vT-model
python3 src/porepy/examples/cold_injection/run_case2b.py -a -j 1.1
echo "--- SIMULATION 3 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2b.py -a -j 1.5
echo "--- SIMULATION 4 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2b.py -a -j 2
echo "--- SIMULATION 5 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2b.py -a -j 2.5
echo "--- SIMULATION 6 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2b.py -a -j 3
echo "--- SIMULATION 7 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2b.py -n -j 3
echo "--- SIMULATION 8 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2b.py -j 3
echo "--- SIMULATION 9 / ${NUM} COMPLETED ---"
# thermal vT-model with uv preconditioning
python3 src/porepy/examples/cold_injection/run_case2c.py -a -j 1.1
echo "--- SIMULATION 10 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2c.py -a -j 1.5
echo "--- SIMULATION 11 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2c.py -a -j 2
echo "--- SIMULATION 12 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2c.py -a -j 2.5
echo "--- SIMULATION 13 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2c.py -a -j 3
echo "--- SIMULATION 14 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2c.py -n -j 3
echo "--- SIMULATION 15 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2c.py -j 3
echo "--- SIMULATION 16 / ${NUM} COMPLETED ---"
# thermal ph-model with uv preconditioning
python3 src/porepy/examples/cold_injection/run_case2d.py -a -j 1.1
echo "--- SIMULATION 17 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2d.py -a -j 1.5
echo "--- SIMULATION 18 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2d.py -a -j 2
echo "--- SIMULATION 19 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2d.py -a -j 2.5
echo "--- SIMULATION 20 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2d.py -a -j 3
echo "--- SIMULATION 21 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2d.py -n -j 3
echo "--- SIMULATION 22 / ${NUM} COMPLETED ---"
python3 src/porepy/examples/cold_injection/run_case2d.py -j 3
echo "--- SIMULATION 23 / ${NUM} COMPLETED ---"
# Plotting and printing results
python3 src/porepy/examples/cold_injection/plot_case2.py
echo "--- RUN CASE 2 COMPLETED ---"