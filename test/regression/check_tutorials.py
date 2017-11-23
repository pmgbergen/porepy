#! /usr/bin/env python
import os, sys, glob


def run_all():
    os.chdir('../../tutorials')
    failed = False
    for file in glob.glob('*.ipynb'):
        new_file = file[:-6] + '.py'
        cmd_convert = 'jupyter nbconvert --to script ' + file
        os.system(cmd_convert)
        remove_plots(new_file)

        cmd_run = 'python ' + str(new_file)
        status = os.system(cmd_run)
        if status != 0:
            print('\n')
            print('*********************\n')
            print(file + ' failed\n\n')
            print('********************\n')
            failed = True
        cmd_delete = 'rm ' + new_file
        os.system(cmd_delete)

    assert not failed

def remove_plots(fn):
    with open(fn) as f:
        content = f.readlines()

    with open(fn, 'w') as f:
        for line in content:
            if line.strip()[:9] != 'plot_grid' and line.strip()[:4] != 'plt.'\
                and line.strip()[:8] != 'frac_viz':
                f.write(line)

run_all()
