#!/bin/bash
# Call the ph flash with the first refinement and minimal local tolerance and test strides.
echo "--- SIMULATION 0 / 22 COMPLETED ---"
python3 run.py -e ph -r 0 -t 7 -s -1 -m 6
echo "--- SIMULATION 1 / 22 COMPLETED ---"
python3 run.py -e ph -r 0 -t 7 -s 1 -m 6
echo "--- SIMULATION 2 / 22 COMPLETED ---"
python3 run.py -e ph -r 0 -t 7 -s 2 -m 6
echo "--- SIMULATION 3 / 22 COMPLETED ---"
python3 run.py -e ph -r 0 -t 7 -s 3 -m 6
echo "--- SIMULATION 4 / 22 COMPLETED ---"
python3 run.py -e ph -r 0 -t 7 -s 4 -m 6
echo "--- SIMULATION 5 / 22 COMPLETED ---"
python3 run.py -e ph -r 0 -t 7 -s 5 -m 6
echo "--- SIMULATION 6 / 22 COMPLETED ---"
# Call the ph flash with optimal stride and test local tolerances.
python3 run.py -e ph -r 0 -t 0 -s 3 -m 6
echo "--- SIMULATION 7 / 22 COMPLETED ---"
python3 run.py -e ph -r 0 -t 1 -s 3 -m 6
echo "--- SIMULATION 8 / 22 COMPLETED ---"
python3 run.py -e ph -r 0 -t 2 -s 3 -m 6
echo "--- SIMULATION 9 / 22 COMPLETED ---"
python3 run.py -e ph -r 0 -t 3 -s 3 -m 6
echo "--- SIMULATION 10 / 22 COMPLETED ---"
python3 run.py -e ph -r 0 -t 4 -s 3 -m 6
echo "--- SIMULATION 11 / 22 COMPLETED ---"
python3 run.py -e ph -r 0 -t 5 -s 3 -m 6
echo "--- SIMULATION 12 / 22 COMPLETED ---"
python3 run.py -e ph -r 0 -t 6 -s 3 -m 6
echo "--- SIMULATION 13 / 22 COMPLETED ---"
# NOTE below config is also run in line 7
# python3 run.py -e ph -r 0 -t 7 -s 3 -m 6
# Run all cases with optimal stride and local tolerance.
python3 run.py -e pT -r 0 -t 2 -s 3 -m 20
echo "--- SIMULATION 14 / 22 COMPLETED ---"
python3 run.py -e ph -r 0 -t 2 -s 3 -m 20
echo "--- SIMULATION 15 / 22 COMPLETED ---"
python3 run.py -e pT -r 1 -t 2 -s 3 -m 20
echo "--- SIMULATION 16 / 22 COMPLETED ---"
python3 run.py -e ph -r 1 -t 2 -s 3 -m 20
echo "--- SIMULATION 17 / 22 COMPLETED ---"
python3 run.py -e pT -r 2 -t 2 -s 3 -m 20
echo "--- SIMULATION 18 / 22 COMPLETED ---"
python3 run.py -e ph -r 2 -t 2 -s 3 -m 20
echo "--- SIMULATION 19 / 22 COMPLETED ---"
#Run most refined cases.
python3 run.py -e ph -r 3 -t 2 -s 3 -m 20
echo "--- SIMULATION 20 / 22 COMPLETED ---"
# Run Simulation for 2D plot with time schedule
python3 run.py -p
echo "--- SIMULATION 21 / 22 COMPLETED ---"
python3 run.py -e pT -r 3 -t 2 -s 3 -m 20
echo "--- SIMULATION 22 / 22 COMPLETED ---"
#Plot results for analysis.
python3 plot.py

