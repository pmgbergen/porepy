#!/bin/bash

python -O main.py
source activate paraview
python post_process.py
tar -czf case3.tar.gz CSV
