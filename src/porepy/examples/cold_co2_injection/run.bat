:: Call the ph flash with the first refinement and minimal local tolerance and test strides.
:: goto :plot
:optimalstride
call python.exe ./run.py -e ph -r 0 -t 7 -s -1 -m 6
call python.exe ./run.py -e ph -r 0 -t 7 -s 1 -m 6
call python.exe ./run.py -e ph -r 0 -t 7 -s 2 -m 6
call python.exe ./run.py -e ph -r 0 -t 7 -s 3 -m 6
call python.exe ./run.py -e ph -r 0 -t 7 -s 4 -m 6
call python.exe ./run.py -e ph -r 0 -t 7 -s 5 -m 6
:: Call the ph flash with optimal stride and test local tolerances.
:optimaltol
call python.exe ./run.py -e ph -r 0 -t 0 -s 3 -m 6
call python.exe ./run.py -e ph -r 0 -t 1 -s 3 -m 6
call python.exe ./run.py -e ph -r 0 -t 2 -s 3 -m 6
call python.exe ./run.py -e ph -r 0 -t 3 -s 3 -m 6
call python.exe ./run.py -e ph -r 0 -t 4 -s 3 -m 6
call python.exe ./run.py -e ph -r 0 -t 5 -s 3 -m 6
call python.exe ./run.py -e ph -r 0 -t 6 -s 3 -m 6
:: NOTE below config is also run in line 7
:: call python.exe ./run.py -e ph -r 0 -t 7 -s 3 -m 6
:: Run all cases with optimal stride and local tolerance.
:comparison
call python.exe ./run.py -e pT -r 0 -t 2 -s 3 -m 24
call python.exe ./run.py -e ph -r 0 -t 2 -s 3 -m 24
call python.exe ./run.py -e pT -r 1 -t 2 -s 3 -m 24
call python.exe ./run.py -e ph -r 1 -t 2 -s 3 -m 24
call python.exe ./run.py -e pT -r 2 -t 2 -s 3 -m 24
call python.exe ./run.py -e ph -r 2 -t 2 -s 3 -m 24
:: Run most refined cases, with pT expected to fail.
:highrefinement
call python.exe ./run.py -e ph -r 3 -t 2 -s 3 -m 24
call python.exe ./run.py -e pT -r 3 -t 2 -s 3 -m 24
:: Run Simulation for 2D plot
:plot2d
call python.exe ./run.py -p
:: Plot results for analysis.
:plot
call python.exe ./plot.py
