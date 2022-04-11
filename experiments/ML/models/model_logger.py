import os

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime

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


def plot_corr_matrix(df, monitoring_params = None, input_params = None):
    # extract correlation matrix. If we have a numpy array, we turn it into a pandas df to
    # extract correlation matrix
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(data = df, columns = monitoring_params + input_params)

    df.drop(labels=['ParameterSet', 'No', 'Time', 'p_low'], axis=1, inplace=True)
    corr = df.corr()
    corr = corr.iloc[12:25, 12:25]

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
        linecolor='k',
        annot=True,
        fmt='.3f',
        cbar_kws={"shrink": .5},
        ax=ax
    )
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right', fontsize=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, horizontalalignment='center', fontsize=20)
    ax.set(title='Correlation matrix')
    plt.tight_layout()
    plt.show()


def log_run(run_name, output_dir,
            mse_vector,
            armsse_vector,
            multi_mse_vector,
            multi_rmse_vector,
            dummy_mse_vector,
            dummy_arrmse_vector,
            prediction_vector,
            runtime_vector,
            seed):

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


def plot_true_pred_dist(prediction_array, output_dir = None, figname = None, run = 0):
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

    # run it in the PyProject folder if you give output_dir as params
    if output_dir:
        os.chdir(output_dir)
        os.makedirs('scatter_plot')
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


# use this for debug
if __name__ == "__main__":
    print()


