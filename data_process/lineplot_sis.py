import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.core.util import event_pb2
import os.path as osp
import tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator
import json

sns.set(style="darkgrid")
SMOOTHFACTOR = 0.1
SMOOTHFACTOR2 = 3
DIV_LINE_WIDTH = 50
txt_store_alg_list = ['CPO', 'PPO-Lagrangian']

def help_func():
    tag2plot = ['episode_return']
    alg_list = ['FSAC'] # 'SAC',
    lbs = ['episode_return'] # 'SAC',
    task = ['Goal_hazards0.15_4']
    palette = "bright"
    goal_perf_list = [-200, -100, -50, -30, -20, -10, -5]
    dir_str = '../data_process/data2plot' # .format(algo name) # /data2plot
    return tag2plot, alg_list, task, lbs, palette, goal_perf_list, dir_str

def plot_eval_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    tag2plot, alg_list, task_list, lbs, palette, _, dir_str = help_func()
    df_dict = {}
    df_in_one_run_of_one_alg = {}
    for task in task_list:
        df_list = []
        for alg in alg_list:
            data2plot_dir = dir_str
            data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
            for num_run, dir in enumerate(data2plot_dirs_list):
                if alg in txt_store_alg_list:
                    eval_dir = data2plot_dir
                    print(eval_dir)
                    df_in_one_run_of_one_alg = get_datasets(eval_dir, tag2plot, alg=alg, num_run=num_run)
                else:
                    eval_dir = data2plot_dir + '/logs/evaluator'
                    print(eval_dir)
                    eval_file = os.path.join(eval_dir,
                                             [file_name for file_name in os.listdir(eval_dir) if file_name.startswith('events')][0])
                    eval_summarys = tf.data.TFRecordDataset([eval_file])
                    data_in_one_run_of_one_alg = {key: [] for key in tag2plot}
                    data_in_one_run_of_one_alg.update({'iteration': []})
                    for eval_summary in eval_summarys:
                        event = event_pb2.Event.FromString(eval_summary.numpy())
                        for v in event.summary.value:
                            t = tf.make_ndarray(v.tensor)
                            for tag in tag2plot:
                                if tag == v.tag[11:]:
                                    data_in_one_run_of_one_alg[tag].append((1-SMOOTHFACTOR)*data_in_one_run_of_one_alg[tag][-1] + SMOOTHFACTOR*float(t)
                                                                           if data_in_one_run_of_one_alg[tag] else float(t))
                                    data_in_one_run_of_one_alg['iteration'].append(int(event.step))
                    len1, len2 = len(data_in_one_run_of_one_alg['iteration']), len(data_in_one_run_of_one_alg[tag2plot[0]])
                    period = int(len1/len2)
                    data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i*period]/10000. for i in range(len2)]

                    data_in_one_run_of_one_alg.update(dict(algorithm=alg, num_run=num_run))
                    df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
                df_list.append(df_in_one_run_of_one_alg)
        total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
        figsize = (6,6)
        axes_size = [0.11, 0.11, 0.89, 0.89] #if env == 'path_tracking_env' else [0.095, 0.11, 0.905, 0.89]
        fontsize = 16
        f1 = plt.figure(1, figsize=figsize)
        ax1 = f1.add_axes(axes_size)
        for tag in tag2plot:
            sns.lineplot(x="iteration", y=tag, hue="algorithm",
                         data=total_dataframe, linewidth=2, palette=palette
                         )
        base = 0
        basescore = sns.lineplot(x=[0., 200.], y=[base, base], linewidth=2, color='black', linestyle='--')
        print(ax1.lines[0].get_data())
        ax1.set_ylabel('')
        ax1.set_xlabel("Iteration [x10000]", fontsize=fontsize)
        handles, labels = ax1.get_legend_handles_labels()
        labels = lbs
        ax1.legend(handles=handles+[basescore.lines[-1]], labels=labels+['Constraint'], loc='upper right', frameon=False, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        # plt.show()
        fig_name = '../data_process/figure/' + task+'-'+tag + '.png'
        plt.savefig(fig_name)
        # allresults = {}
        # results2print = {}
        #
        # for alg, group in total_dataframe.groupby('algorithm'):
        #     allresults.update({alg: []})
        #     for ite, group1 in group.groupby('iteration'):
        #         mean = group1['episode_return'].mean()
        #         std = group1['episode_return'].std()
        #         allresults[alg].append((mean, std))
        #
        # for alg, result in allresults.items():
        #     mean, std = sorted(result, key=lambda x: x[0])[-1]
        #     results2print.update({alg: [mean, 2 * std]})
        #
        # print(results2print)


def get_datasets(logdir, tag2plot, alg, condition=None, smooth=SMOOTHFACTOR2, num_run=0):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.
    """
    # global exp_idx
    # global units
    datasets = []

    for root, _, files in os.walk(logdir):

        if 'progress.txt' in files:
            try:
                exp_data = pd.read_table(os.path.join(root,'progress.txt'))
            except:
                print('Could not read from %s'%os.path.join(root,'progress.txt'))
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns),'Performance',exp_data[performance])
            exp_data.insert(len(exp_data.columns),'algorithm',alg)
            exp_data.insert(len(exp_data.columns), 'iteration', exp_data['TotalEnvInteracts']/10000)
            exp_data.insert(len(exp_data.columns), 'episode_cost', exp_data['AverageEpCost'])
            exp_data.insert(len(exp_data.columns), 'num_run', num_run)
            datasets.append(exp_data)
            data = datasets

            for tag in tag2plot:
                if smooth > 1:
                    """
                    smooth data with moving window average.
                    that is,
                        smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
                    where the "smooth" param is width of that window (2k+1)
                    """
                    y = np.ones(smooth)
                    for datum in data:
                        x = np.asarray(datum[tag])
                        z = np.ones(len(x))
                        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
                        datum[tag] = smoothed_x

            if isinstance(data, list):
                data = pd.concat(data, ignore_index=True)

            slice_list = tag2plot + ['algorithm', 'iteration', 'num_run']

    return data.loc[:, slice_list]

def plot_phi_curve_in_single_run(log):
    data = np.load(log, allow_pickle=True)
    df_list = []
    for idx in range(data.shape[0]):
        data_dict = data[idx]
        data_dict.update(dict(step = np.arange(110), run=idx))
        df_list.append(pd.DataFrame(data[idx]))
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    figsize = (9, 3)
    axes_size = [0.05, 0.22, 0.93, 0.75]  # if env == 'path_tracking_env' else [0.095, 0.11, 0.905, 0.89]
    fontsize = 16
    f1 = plt.figure(1, figsize=figsize)
    ax1 = f1.add_axes(axes_size)
    # for tag in tag2plot:
    sns.lineplot(x="step", y='phi_list', hue="run",
                 data=total_dataframe, linewidth=2, palette = "bright"
                 )
    base = 0
    basescore = sns.lineplot(x=[0., 110.], y=[base, base], linewidth=2, color='black', linestyle='--')
    ax1.set_ylabel('')
    ax1.set_xlabel("Step", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    # plt.show()
    fig_name = '../data_process/figure/phi.png'
    plt.savefig(fig_name)






if __name__ == '__main__':
    # env = 'inverted_pendulum_env'  # inverted_pendulum_env path_tracking_env
    # plot_eval_results_of_all_alg_n_runs()
    plot_phi_curve_in_single_run('../results/FSAC-A/Unicycle/data2plot/Unicycle-2021-07-28-18-27-45/logs/evaluator/n_metrics_list_ite630000.npy')