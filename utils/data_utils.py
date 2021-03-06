import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np


def make_labelled_dataset(run_label, run_data):
    '''
        input: xml config file, csv/xlsx output generated by aplysia
        output: pandas dataframe with all the run_history and its label

    '''

    if run_data[-4:] == '.csv' or run_data[-4:] == '.CSV':
        df = pd.read_csv(run_data, delimiter=';')
    else:
        df = pd.read_excel(run_data)

    # delete useless rows/columns
    df.drop(labels='Unnamed: 0', axis=1, inplace=True)
    df.dropna(inplace=True)

    # check if all parameter set has been captured correcty
    parameter_set_counts = list(df['ParameterSet'].value_counts())
    if len(set(parameter_set_counts)) != 1:
        raise Exception("simulations does not have the same number of samples. ")

        # convert all datatypes to numeric (are inferred as object from pandas)
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', '.'), errors='coerce'))

    # retrieve and transform the xml file in a pandas df
    input_tree = ET.parse(run_label)
    root = input_tree.getroot()
    root.remove(root[0])  # we remove the TimeInterval info we do not need to retrieve it

    header = []
    header_ext = []
    n_runs = len(set(df['ParameterSet']))
    run_labels = {run_index: [] for run_index in range(1, n_runs + 1)}

    # we need to replace comma to dot in order to read labels as float.
    for child in root:
        header.append(child[1].text)
        header_ext.append(child[0].text)
        for i in range(1, n_runs + 1):
            run_labels[i].append(float(child[i + 1].text.replace(',', '.')))

    # create a df just with the set of parameters in input
    run_labels_df = pd.DataFrame.from_dict(run_labels, orient='index', columns=header)
    run_labels_df['ParameterSet'] = run_labels_df.index
    run_labels_df = run_labels_df.astype(float)

    labelled_df = pd.merge(df, run_labels_df, left_on='ParameterSet', right_on='ParameterSet')

    return labelled_df


def read_merged_dataset(merged_df_path):
    '''
        input: xml config file, csv/xlsx output generated by aplysia
        output: pandas dataframe with all the run_history and its label

    '''

    if merged_df_path[-4:] == '.csv' or merged_df_path[-4:] == '.CSV':
        df = pd.read_csv(merged_df_path, delimiter=',')
    else:
        df = pd.read_excel(merged_df_path)

    # delete useless rows/columns generated from the excel format
    df.drop(labels='Unnamed: 0.1', axis=1, inplace=True)
    df.drop(labels='Unnamed: 0', axis=1, inplace=True)
    df.dropna(inplace=True)

    # check if all parameter set has been captured correcty
    parameter_set_counts = list(df['ParameterSet'].value_counts())
    if len(set(parameter_set_counts)) != 1:
        raise Exception("simulations does not have the same number of samples. ")

    return df

# check if we have a difference in time using the Time column
def time_diff_check(df):
    '''
        input: df with data (Time and ParameterSet column are needed)
        output: vector with the difference in terms of time step

    '''
    timestamp = df['Time']
    diff = []
    for i in range(0, len(set(df['ParameterSet'])) - 1):
        diff.append(timestamp[1000 + (i * 1000)] - timestamp[999 + (i * 1000)])
    return diff