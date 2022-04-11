import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from data_utils import make_labelled_dataset
import os

import matplotlib.pyplot as plt

# The following makes the plot look nice
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)


if __name__ == '__main__':
    os.chdir("C:/Users/Prestige/Desktop/Paolo/UNi/ERASMUS/TESI/OneDrive - KTH/Aplysia/exctracted_data/new release/output/PyProject/")

    csv_input_path = './output csv/run2_bounded.xlsx'
    xml_input_path = './input xml/bounded run_history/run_2_bounded.xml'

    df = make_labelled_dataset(xml_input_path, csv_input_path)

    # uncomment what is needed to be plotted
    subset = ['ParameterSet',
              # 'No',
              'Time',
              'CycleTime',
              # 'Heart rate',
              # 'Systemic arterial pressure',
              # 'Pulmonary arterial pressure',
              'Right atrial pressure',
              'Left atrial pressure',
              'Cardiac output/venous return',
              'Left ventricular ejection fraction',
              'Right ventricular ejection fraction',
              # 'Hemoglobin',
              # 'Systemic arterial oxygen saturation',
              'Mixed venous oxygen saturation'
              ]

    df[subset].plot(subplots=True, figsize=(22, 4 * len(subset)))
    plt.rcParams['axes.grid'] = True
    plt.show()
