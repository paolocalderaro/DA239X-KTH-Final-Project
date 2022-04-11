from utils.data_utils import make_labelled_dataset
import pandas as pd
import numpy as np
import sys
import os
"""
    arguments: 
        xlsx run_history directory path (from the cwd)
        xlm label directory path
        merged data directory path
        .txt with already merged run_history
"""
if __name__ == '__main__':

    cwd = os.getcwd() # run_history this script in PyProject

    run_dir = os.path.join(cwd,sys.argv[1])
    label_dir = os.path.join(cwd,sys.argv[2])
    merged_dir = os.path.join(cwd,sys.argv[3])


    # set here if we want to check in the txt file the already merged runs
    merged_check = True
    if merged_check:
        already_merged_txt = os.path.join(cwd,sys.argv[4])

    df_list = []
    run_count = 0

    for previous_run in os.listdir(merged_dir):
        existing_df = pd.read_csv(os.path.join(merged_dir, previous_run), delimiter=',')

        # update the current number of run_history
        run_count += existing_df['ParameterSet'].max()

        # store the df
        df_list.append(existing_df)


    # build list of run_history names to merge.
    # We check if a run_history was already merged and if we have actual data (not .txt file)
    runs_to_merge = []
    for run_file in os.listdir(run_dir):
        # if we have already red it, break
        if merged_check:
            with open(already_merged_txt, 'r') as f:
                if run_file in f.read():
                    continue

        if run_file[-4:] != '.txt':
            runs_to_merge.append(run_file)


    for run in runs_to_merge:

        # retrieve the run_history from the name of run_history file:
        label_run = run.split('-')[1][:-5] + '.xml'
        current_df = make_labelled_dataset( os.path.join(label_dir, label_run),os.path.join(run_dir,run))

        # update the id by increasing of the same amount as the run_history encountered
        current_df['ParameterSet'] += run_count

        # update the current number of run_history
        run_count += current_df['ParameterSet'].nunique()

        # store the df
        df_list.append(current_df)

        if merged_check:
            with open(already_merged_txt, 'a') as f:
                f.write(run)
                f.write("\n")


    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_csv(os.path.join(merged_dir,'merged.csv'))