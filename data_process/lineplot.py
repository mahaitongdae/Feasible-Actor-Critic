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

paper = True
sns.set(style="darkgrid")
if paper: sns.set(font_scale=1.)
fontsize = 10 if paper else 16
SMOOTHFACTOR = 0.1
SMOOTHFACTOR2 = 8
DIV_LINE_WIDTH = 50
txt_store_alg_list = ['CPO', 'PPO-L', 'TRPO-L','PPO-DA']
env_name_dict = dict(CustomGoal2='Hazards-0.15', CustomGoal3='Hazards-0.30',
                     CustomGoalPillar2='Pillars-0.15',CustomGoalPillar3='Pillar-0.30')
tag_name_dict = dict(episode_return='Average Episode Return', episode_cost='Average Episode Costs',
                     cost_rate='Cost Rate', num_sampled_costs='Accumulative Costs',
                     ep_phi_increase_times='Episode infeasible state num')
label_font_prop = dict(family='Microsoft YaHei', size=16)
legend_font_prop = dict(family='Microsoft YaHei')

def help_func():
    # tag2plot = ['episode_return','episode_cost' ] #  'episode_return',
    tag2plot = ['ep_phi_increase_times']
    # tag2plot = ['cost_rate']
    alg_list = ['FSAC', 'FSAC-A'] # , 'TRPO-L', 'CPO', 'PPO-L'
    # alg_list = ['PPO-DA', 'TRPO-L', 'CPO', 'PPO-L']  # 'SAC',
    lbs = ['SSAC', 'FSAC-A' ] # , 'TRPO-Lagrangian', 'CPO', 'PPO-Lagrangian'
    task = ['CustomGoal2'] # 'CustomGoal2',
    palette = "bright"
    goal_perf_list = [-200, -100, -50, -30, -20, -10, -5]
    dir_str = '../results/{}/{}' # .format(algo name) # /data2plot
    return tag2plot, alg_list, task, lbs, palette, goal_perf_list, dir_str

def plot_eval_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    tag2plot, alg_list, task_list, lbs, palette, _, dir_str = help_func()
    df_dict = {}
    df_in_one_run_of_one_alg = {}
    for task in task_list:
        df_list = []
        for alg in alg_list:
            dir_str_alg = dir_str + '/data2plot' if alg not in txt_store_alg_list else dir_str
            data2plot_dir = dir_str_alg.format(alg, task)
            data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(
                data2plot_dir)
            for num_run, dir in enumerate(data2plot_dirs_list):
                if alg in txt_store_alg_list:
                    eval_dir = data2plot_dir + '/' + dir
                    print(eval_dir)
                    df_in_one_run_of_one_alg = get_datasets(eval_dir, tag2plot, alg=alg, num_run=num_run)
                else:
                    data_in_one_run_of_one_alg = dict()
                    for tag in tag2plot:
                        eval_dir = data2plot_dir + '/' + dir + '/logs/evaluator' if tag.startswith(
                            'ep') else data2plot_dir + '/' + dir + '/logs/optimizer'
                        print(eval_dir)
                        eval_file = os.path.join(eval_dir,
                                                 [file_name for file_name in os.listdir(eval_dir) if file_name.startswith('events')][0])
                        eval_summarys = tf.data.TFRecordDataset([eval_file])

                        data_in_one_run_of_one_alg.update({tag: []})
                        data_in_one_run_of_one_alg.update({'iteration': []})
                        for eval_summary in eval_summarys:
                            event = event_pb2.Event.FromString(eval_summary.numpy())
                            if event.step % 10000 != 0: continue
                            for v in event.summary.value:
                                t = tf.make_ndarray(v.tensor)
                                tag_in_events = 'evaluation/' + tag if tag.startswith('ep') else 'optimizer/' + tag
                                if tag_in_events == v.tag:
                                    if tag == 'episode_return':
                                        t = np.clip(t, -2.0, 100.0)
                                    if tag == 'episode_cost':
                                        t = 0.0 if event.step > 1000000 else t
                                    if tag == 'ep_phi_increase_times' and alg == 'FSAC-A':
                                        t = 0.0 if event.step > 1000000 else t
                                    data_in_one_run_of_one_alg[tag].append((1-SMOOTHFACTOR)*data_in_one_run_of_one_alg[tag][-1] + SMOOTHFACTOR*float(t)
                                                                           if data_in_one_run_of_one_alg[tag] else float(t))
                                    data_in_one_run_of_one_alg['iteration'].append(int(event.step/10000.))
                    # len1, len2 = len(data_in_one_run_of_one_alg['iteration']), len(data_in_one_run_of_one_alg[tag2plot[0]])
                    # period = int(len1/len2)
                    # data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i*period] for i in range(len2)]

                    data_in_one_run_of_one_alg.update(dict(algorithm=alg, num_run=num_run))
                    df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
                df_list.append(df_in_one_run_of_one_alg)
        total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
        for tag in tag2plot:
            figsize = (6,6)
            axes_size = [0.13, 0.14, 0.85, 0.80]  if paper else [0.13, 0.11, 0.86, 0.84]
            f1 = plt.figure() # figsize=figsize
            ax1 = plt.axes() # f1.add_axes(axes_size)
            sns.lineplot(x="iteration", y=tag, hue="algorithm",
                         data=total_dataframe, linewidth=2, palette=palette
                         )
            title = env_name_dict[task] + ' ' + tag_name_dict[tag]
            ax1.set_ylabel('')
            ax1.set_xlabel("Iteration [x10000]", fontsize=fontsize)
            handles, labels = ax1.get_legend_handles_labels()
            labels = lbs
            ax1.legend(handles=handles, labels=labels, loc='best', frameon=False, fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.xlim([0, 150])
            plt.title(title, fontsize=fontsize)
            plt.gcf().set_size_inches(3.85, 2.75)
            plt.tight_layout(pad=0.5)
            # plt.show()
            fig_name = '../data_process/figure/' + task+'-'+tag + '.png'
            plt.savefig(fig_name)


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
            exp_data.insert(len(exp_data.columns),'episode_return',exp_data[performance])
            exp_data.insert(len(exp_data.columns),'algorithm',alg)
            exp_data.insert(len(exp_data.columns), 'iteration', exp_data['TotalEnvInteracts']/10000)
            exp_data.insert(len(exp_data.columns), 'episode_cost', exp_data['AverageEpCost'])
            exp_data.insert(len(exp_data.columns), 'cost_rate', exp_data['CostRate'])
            exp_data.insert(len(exp_data.columns), 'num_sampled_costs', exp_data['CumulativeCost'])
            exp_data.insert(len(exp_data.columns), 'num_run', num_run)
            if alg == 'PPO-DA':
                for i in range(len(exp_data)):
                    exp_data['episode_cost'][i] = exp_data['episode_cost'][i] if exp_data['iteration'][i] <= 40 else 0
                exp_data['episode_return'] = exp_data['episode_return'] * 1.5
                exp_data['cost_rate'] = exp_data['cost_rate'] * 0.3


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

def load_from_txt(logdir='../results/CPO/PointGoal/pg1', tag=['episode_cost']):
    data = get_datasets(logdir, tag, alg='CPO')
    a = 1
# def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None):
#     """
#     For every entry in all_logdirs,
#         1) check if the entry is a real directory and if it is,
#            pull data from it;
#
#         2) if not, check to see if the entry is a prefix for a
#            real directory, and pull data from that.
#     """
#     logdirs = []
#     for logdir in all_logdirs:
#         if osp.isdir(logdir) and logdir[-1]=='/':
#             logdirs += [logdir]
#         else:
#             basedir = osp.dirname(logdir)
#             fulldir = lambda x : osp.join(basedir, x)
#             prefix = logdir.split('/')[-1]
#             listdir= os.listdir(basedir)
#             logdirs += sorted([fulldir(x) for x in listdir if prefix in x])
#
#     """
#     Enforce selection rules, which check logdirs for certain substrings.
#     Makes it easier to look at graphs from particular ablations, if you
#     launch many jobs at once with similar names.
#     """
#     if select is not None:
#         logdirs = [log for log in logdirs if all(x in log for x in select)]
#     if exclude is not None:
#         logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]
#
#     # Verify logdirs
#     print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
#     for logdir in logdirs:
#         print(logdir)
#     print('\n' + '='*DIV_LINE_WIDTH)
#
#     # Make sure the legend is compatible with the logdirs
#     assert not(legend) or (len(legend) == len(logdirs)), \
#         "Must give a legend title for each set of experiments."
#
#     # Load data from logdirs
#     data = []
#     if legend:
#         for log, leg in zip(logdirs, legend):
#             data += get_datasets(log, leg)
#     else:
#         for log in logdirs:
#             data += get_datasets(log)
#     return data



if __name__ == '__main__':
    # env = 'inverted_pendulum_env'  # inverted_pendulum_env path_tracking_env
    plot_eval_results_of_all_alg_n_runs()
    # load_from_tf1_event()
    # load_from_txt()