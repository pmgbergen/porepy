#!/bin/bash

python -O main.py
source activate paraview
python post_process.py
tar -czf case1.tar.gz CSV
