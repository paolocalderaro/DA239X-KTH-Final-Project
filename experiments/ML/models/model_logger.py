import os

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from sklearn.model_selection import train_test_split

from utils.data_utils import read_merged_dataset


def save_run_meta():
    return


def save_trained_model():
    return


def save_run(model,output_dir):
    return


def prediction_plotter_2d(y_true, y_pred, output_dir = None, figname= None):
    if output_dir:
       os.chdir(output_dir)

    params = {'legend.fontsize': 'xx-large',
              'figure.figsize': (9, 6),
              'axes.labelsize': 'xx-large',
              'axes.titlesize': 'xx-large',
              'xtick.labelsize': 'xx-large',
              'ytick.labelsize': 'xx-large'}
    plt.rcParams.update(params)

    tsne = TSNE(random_state=1, n_iter=15000, metric="cosine", init='pca')
    true_embs = tsne.fit_transform(y_true)
    pred_embs = tsne.fit_transform(y_pred)

    sns.set_style("white")
    sns.color_palette("icefire", as_cmap=True)

    sns.scatterplot(true_embs[:, 0], true_embs[:, 1], label='True value', marker= '^', edgecolor="k", s=50)
    sns.scatterplot(pred_embs[:, 0], pred_embs[:, 1], marker='X', edgecolor="k", s=50, label='Predicted value', color='red')
    plt.title('2D visualization using t-SNE')
    plt.legend()
    if output_dir:
        plt.savefig(figname +'.svg',
                    dpi=300,
                    format='svg',
                    bbox_inches='tight'
                    )
    plt.show()


def plot_corr_matrix(df, dataset_path = None, monitoring_params = None, input_params = None):
    '''
    :param df: dataframe containing the data
    :param dataset_path: path of the dataset
    :param monitoring_params: subset of params
    :param input_params: subset of params
    :return:
    extract correlation matrix. It also order the matrix using the cuthill mckee algorithm.
    If we have a numpy array, we turn it into a pandas df to extract correlation matrix.
    '''

    if isinstance(df, np.ndarray):
        df = pd.DataFrame(data=df, columns=monitoring_params + input_params)

    df = read_merged_dataset(dataset_path)
    monitoring_params = [  # 'ParameterSet',
        #'No',
        # 'Time',
        'CycleTime',
        'Heart rate',
        'Systemic arterial pressure',
        'Pulmonary arterial pressure',
        'Right atrial pressure',
        'Left atrial pressure',
        'Cardiac output/venous return',
        'Left ventricular ejection fraction',
        'Right ventricular ejection fraction',
        'Hemoglobin',
        'Systemic arterial oxygen saturation',
        'Mixed venous oxygen saturation']
    input_params = [#'ParameterSet',
                    # 'HR',
                    'TotalVascularVolume',
                    'e_lvmax',
                    'e0_lv',
                    'e_rvmax',
                    'e0_rv',
                    'SVR',
                    'PVR',
                    'Ea',
                    'Epa',
                    # 'Hb',
                    'O2Cons',
                    'PulmShuntFraction'
                    # 'p_low'
                    ]
    df.drop(labels=['ParameterSet', 'No', 'Time', 'p_low'], axis=1, inplace=True)
    df = df[monitoring_params + input_params]
    corr = df.corr()

    # matrix ordering
    corr_np = csr_matrix(corr.to_numpy())
    perm = reverse_cuthill_mckee(corr_np, symmetric_mode=True)
    for i in range(len(perm)):
        corr_np[:, i] = corr_np[perm, i]
    for i in range(len(perm)):
        corr_np[i, :] = corr_np[i, perm]

    corr = pd.DataFrame(data=corr_np.toarray(), columns=monitoring_params + input_params)


    # reset the index for the rows
    corr.set_index(pd.Series(monitoring_params + input_params), inplace=True)

    # slice the index (useful to plot just some parameters)
    # [0:11, 0:11] for input params
    # [12:23, 12:23] for output params
    corr = corr.iloc[0:11, 12:23]

    #mask
    #mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.set(font_scale=1.3)
    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    sns.heatmap(
        data=corr,
        vmin=-1.0,
        vmax=1.0,
        center=0,
        cmap=cmap,
        square=True,
        linewidths=0.5,
        linecolor='w',
        annot=True,
        fmt='.3f',
        cbar_kws={"shrink": .5},
        #mask=mask,
        ax=ax
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, horizontalalignment='right', fontsize=20)
    ax.set(title='Correlation matrix')
    plt.tight_layout()
    plt.show()


def log_run(run_name, output_dir,mse_vector,armsse_vector,multi_mse_vector,multi_rmse_vector,dummy_mse_vector,
            dummy_arrmse_vector,prediction_vector,runtime_vector,seed):
    '''
    :param run_name:
    :param output_dir:
    :param mse_vector:
    :param armsse_vector:
    :param multi_mse_vector:
    :param multi_rmse_vector:
    :param dummy_mse_vector:
    :param dummy_arrmse_vector:
    :param prediction_vector:
    :param runtime_vector:
    :param seed:
    :return:
    '''

    # save in the choosen output dir
    #os.chdir(output_dir)
    ts = datetime.now()
    filename = str(ts.timestamp()) +'__log.txt'

    ts = ts.strftime("%d/%m/%Y %H:%M:%S")

    f = open(filename, "w+")
    f.write('-------------------------------')
    f.write('\n')
    f.write('seed list: ')
    f.write('\n')
    f.write(str(seed))
    f.write('\n')
    f.write('\n')

    f.write('-------------------------------')
    f.write('\n')
    f.write('time stamp: ')
    f.write('\n')
    f.write(str(ts))
    f.write('\n')
    f.write('\n')

    f.write('-------------------------------')
    f.write('\n')
    f.write('MSE for each run: ')
    f.write('\n')
    f.write(str(mse_vector))
    f.write('\n')
    f.write('\n')

    f.write('-------------------------------')
    f.write('\n')
    f.write('MSE for each run (dummy) : ')
    f.write('\n')
    f.write(str(dummy_mse_vector))
    f.write('\n')
    f.write('\n')

    f.write('-------------------------------')
    f.write('\n')
    f.write('aRMSSE for each run: ')
    f.write('\n')
    f.write(str(armsse_vector))
    f.write('\n')
    f.write('\n')

    f.write('-------------------------------')
    f.write('\n')
    f.write('aRRMSE for each run (dummy): ')
    f.write('\n')
    f.write(str(dummy_arrmse_vector))
    f.write('\n')
    f.write('\n')

    f.write('-------------------------------')
    f.write('\n')
    f.write('Multi MSE for each run: ')
    f.write('\n')
    f.write(str(multi_mse_vector))
    f.write('\n')
    f.write('\n')

    f.write('-------------------------------')
    f.write('\n')
    f.write('multi RMSE for each run: ')
    f.write('\n')
    f.write(str(multi_rmse_vector))
    f.write('\n')
    f.write('\n')

    f.write('-------------------------------')
    f.write('\n')
    f.write('runtime: ')
    f.write('\n')
    f.write(str(runtime_vector))
    f.write('\n')
    f.write('\n')


    f.write('-------------------------------')
    np.save('run__pred_vector.npy',prediction_vector,allow_pickle=True)


    f.close()


def plot_true_pred_dist(prediction_array, output_dir = None, run = 0):
    '''
    :param prediction_array: the array saved from the ML runs
    :return: plot in std_output the distribution of the target variables.
            save the fig in output_dir, if given.
    '''

    if output_dir:
        os.chdir(output_dir)
        os.makedirs('dist_plot')
        os.chdir('dist_plot')

    labels = ['TotalVascularVolume',
              'e_lvmax',
              'e0_lv',
              'e_rvmax',
              'e0_rv',
              'SVR',
              'PVR',
              'Ea',
              'Epa',
              'O2Cons',
              'PulmShuntFraction'
              ]
    y_true = prediction_array[run][0]
    y_pred = prediction_array[run][1]

    for attribute in range(len(y_true[0])):
        attr_name = labels[attribute]
        current_df = pd.DataFrame(data=np.c_[y_true[:, attribute], y_pred[:, attribute]],
                                  columns=['True', 'Predicted'])
        sns.displot(data=current_df, kind='kde', fill=True, palette=sns.color_palette('bright')[:2])
        plt.title(attr_name + ' distribution  (test data).')
        plt.tight_layout()
        if output_dir:
            plt.savefig(attr_name+'.svg',
                        dpi=300,
                        format='svg',
                    bbox_inches='tight'
                    )
        plt.show()


def plot_true_pred_scatter(prediction_array,prediction_array_path = None ,output_dir = None, run = 0):
    '''
        :param prediction_array: array containing the prediction
        :param seed: seed used for the run to analyze
        :param prediction_array_path: path in which the prediction array is located
        :param output_dir: directory in which to save the .svg plot
        :param run: number of the run to analyze
        :return: print on std output the plot

        scatter plot displaying the predicted values on the y-axis and the true value on the x-axis.
        the red line plotted is the bisector so it describes the ideal case in which predicted = true for all data point.
    '''

    os.chdir(output_dir)
    # run it in the PyProject folder if you give output_dir as params
    if output_dir and (not os.path.isdir('./scatter_plot')):
        os.makedirs('scatter_plot')
        os.chdir('./scatter_plot')

    if os.path.isdir('./scatter_plot'):
        os.chdir('./scatter_plot')

    if prediction_array_path is not None:
        prediction_array = np.load(prediction_array_path)

    labels = ['TotalVascularVolume',
              'e_lvmax',
              'e0_lv',
              'e_rvmax',
              'e0_rv',
              'SVR',
              'PVR',
              'Ea',
              'Epa',
              'O2Cons',
              'PulmShuntFraction'
              ]

    y_true = prediction_array[run][0]
    y_pred = prediction_array[run][1]

    for attribute in range(len(y_true[0])):
        attr_name = labels[attribute]
        sns.relplot(x=y_true[:,attribute], y=y_pred[:,attribute], marker='X', edgecolor="k", s=50)
        x_ideal = np.linspace(y_true[:,attribute].min() , y_true[:,attribute].max(), 200)
        y_ideal = x_ideal
        plt.plot(x_ideal,y_ideal, color='r')
        plt.title(attr_name)
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.tight_layout()
        if output_dir:
            plt.savefig(attr_name + '__scatter.svg',
                            dpi=300,
                            format='svg',
                            bbox_inches='tight'
                            )
        plt.show()


def plot_true_pred_color_code_scatter(prediction_array, seed, prediction_array_path = None ,output_dir = None, run = 0):
    '''
        :param prediction_array: array containing the prediction
        :param seed: seed used for the run to analyze
        :param prediction_array_path: path in which the prediction array is located
        :param output_dir: directory in which to save the .svg plot
        :param run: number of the run to analyze
        :return: print on std output the plot

        here we color map the entry on the scatter plot depending on the type of patient which they belong.
    '''
    if prediction_array_path is not None:
        prediction_array = np.load(prediction_array_path)

    os.chdir(output_dir)
    if output_dir and (not os.path.isdir('./scatter_plot_cm')):
        os.makedirs('scatter_plot_cm')
        os.chdir('./scatter_plot_cm')

    if os.path.isdir('./scatter_plot_cm'):
        os.chdir('./scatter_plot_cm')

    run_dictionary = {
        '0': 'Adult 20y',
        '1': 'Adult 60y',
        '2': 'Adult 80y',
        '3': 'Adult 20y',
        '4': 'Adult 40y',
        '5': 'Adult 60y',
        '6': 'Adult 80y',
        '7': 'Biventricular Failure',
        '8': 'Left ventricular systolic failure',
        '9': 'Stiff ventricle',
        '10': 'Relaxation abnormalilty',
        '11': 'Right heart failure'
    }

    # create an array that resemble the structure of the dataset before the split:
    # - first 20 entries as case 1
    # - other 20 entries as case 2 and so on.
    label_list = []
    X = np.random.rand(240)
    for i in range(12):
        for _ in range(20):
            label_list.append(i)

    # reproduce the split used in the run
    np.random.seed(seed)  # seed for reproducibility
    _, _, _, run_label = train_test_split(X, label_list, train_size=0.8)

    labels = ['TotalVascularVolume',
              'e_lvmax',
              'e0_lv',
              'e_rvmax',
              'e0_rv',
              'SVR',
              'PVR',
              'Ea',
              'Epa',
              'O2Cons',
              'PulmShuntFraction'
              ]

    y_true = prediction_array[run][0]
    y_pred = prediction_array[run][1]

    for attribute in range(len(y_true[0])):
        attr_name = labels[attribute]

        y_true_current = y_true[:, attribute]
        y_pred_current = y_pred[:, attribute]

        # make a df with the labels info
        current_df = pd.DataFrame(list(zip(y_true_current, y_pred_current, run_label)),
                                  columns=['y_true', 'y_pred', 'physiology'])

        # manage the run label column. It needs to be string type
        current_df['physiology'] = current_df['physiology'].apply(lambda x: str(x))
        current_df.replace({'physiology': run_dictionary}, inplace=True)

        g = sns.relplot(data=current_df, x="y_true", y="y_pred", hue="physiology", marker='o', edgecolor="k", s=40)
        x_ideal = np.linspace(y_true[:, attribute].min(), y_true[:, attribute].max(), 200)
        y_ideal = x_ideal
        plt.plot(x_ideal, y_ideal, color='r')
        g._legend.remove()
        plt.legend(bbox_to_anchor=(1.05, 0.8), loc='upper left', borderaxespad=0)
        plt.title(attr_name)
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.tight_layout()
        if output_dir:
            plt.savefig(attr_name + '__scatter__cm.svg',
                        dpi=300,
                        format='svg',
                        bbox_inches='tight'
                        )
        plt.show()






    return

# use this for debug
if __name__ == "__main__":
    seed = 26
    run = 0
    prediction_array = np.load('experiments/ML/run_history/MultiOutputRegressor(GradientBoostingRegressor(random_state=seed, max_depth = 3))/run__pred_vector.npy')

    output_dir = 'experiments/ML/run_history/MultiOutputRegressor(GradientBoostingRegressor(random_state=seed, max_depth = 3))/scatter_plot_cm/'
    run_dictionary = {
        '0': 'Adult 20y',
        '1': 'Adult 60y',
        '2': 'Adult 80y',
        '3': 'Adult 20y',
        '4': 'Adult 40y',
        '5': 'Adult 60y',
        '6': 'Adult 80y',
        '7': 'Biventricular Failure',
        '8': 'Left ventricular systolic failure',
        '9': 'Stiff ventricle',
        '10': 'Relaxation abnormalilty',
        '11': 'Right heart failure'
    }

    # create an array that resemble the structure of the dataset before the split:
    # - first 20 entries as case 1
    # - other 20 entries as case 2 and so on.
    label_list = []
    X = np.random.rand(240)
    for i in range(12):
        for _ in range(20):
            label_list.append(i)

    # reproduce the split used in the run
    np.random.seed(seed)  # seed for reproducibility
    _, _, _, run_label = train_test_split(X, label_list, train_size=0.8)

    labels = ['TotalVascularVolume',
              'e_lvmax',
              'e0_lv',
              'e_rvmax',
              'e0_rv',
              'SVR',
              'PVR',
              'Ea',
              'Epa',
              'O2Cons',
              'PulmShuntFraction'
              ]

    y_true = prediction_array[run][0]
    y_pred = prediction_array[run][1]

    for attribute in range(len(y_true[0])):
        attr_name = labels[attribute]

        y_true_current = y_true[:, attribute]
        y_pred_current = y_pred[:, attribute]

        # make a df with the labels info
        current_df = pd.DataFrame(list(zip(y_true_current, y_pred_current, run_label)),
                                  columns=['y_true', 'y_pred', 'physiology'])

        # manage the run label column. It needs to be string type
        current_df['physiology'] = current_df['physiology'].apply(lambda x: str(x))
        current_df.replace({'physiology': run_dictionary}, inplace=True)

        g = sns.relplot(data=current_df, x="y_true", y="y_pred", hue="physiology", marker='o', edgecolor="k", s=40)
        x_ideal = np.linspace(y_true[:, attribute].min(), y_true[:, attribute].max(), 200)
        y_ideal = x_ideal
        plt.plot(x_ideal, y_ideal, color='r')
        g._legend.remove()
        plt.legend(bbox_to_anchor=(1.05, 0.8), loc='upper left', borderaxespad=0)
        plt.title(attr_name)
        plt.xlabel('True value')
        plt.ylabel('Predicted value')
        plt.tight_layout()
        os.chdir(output_dir)
        plt.savefig(attr_name + '__scatter__cm.svg',
                        dpi=300,
                        format='svg',
                        bbox_inches='tight'
                        )
        plt.show()











