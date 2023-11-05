import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
import argparse
import os
from parse_grid_results import get_result_df, get_best_all_models, add_metric_column, simple_metric, \
    representation_metric, representation_metric_ndcg, representation_metric_rnd, acc_metric, \
    representation_metric_acc, all_measures_metric, all_measures_weighted_metric, wiki_metric, get_repr_csv, gpa_weighted_metric
from constants import dataset_names, name_map, name_csv, name_2d, colors, colors_points


def map_name(str):
    s = "{modeltype} {hasnoise}"
    if 'nonoise' in str:
        s.format(modeltype="{modeltype}", hasnoise="w/o noise")
    else:
        s.format(modeltype="{modeltype}", hasnoise="w. noise")
    if 'advflip' in str.lower():
        s.format(modeltype="Adversarial, Symmetric")
    elif 'adv' in str.lower():
        s.format(modeltype="Adversarial")
    elif 'sym' in str.lower():
        s.format(modeltype="Symmetric")
    elif 'list' in str.lower():
        s.format(modeltype="DELTR")


def plot_opti(x_list, y_list, ax, color, style):
    """
    Plots the pareto optimal line
    :param x_list: array of x coordinates
    :param y_list: array of y coordinates
    :param ax:     ax of the plot
    :param color:  color of the line
    :param style:  style of the line
    :return: 0
    """
    points = np.transpose([x_list, y_list])

    big_p = [0, 0]
    for p in points:
        max_old = big_p[0] ** 2 + big_p[1] ** 2
        max_cur = p[0] ** 2 + p[1] ** 2
        if max_cur > max_old:
            big_p = [p[0], p[1]]

    x_vals = np.array(ax.get_xlim())
    y_vals = big_p[0] + big_p[1] - 1 * x_vals
    plt.plot(x_vals, y_vals, style, color=color, zorder=1)
    return 0


def plot_2d(x, y, model_names, title="all", eval_ana_name=None, width_fig=2.35, height=2, xlabel=None, ylabel=None,
            msize=10, only_model="ADV_FFDR"):
    """
    Plots the 2d scatter plots for a dataset and the optimal line
    """
    fig, ax = plt.subplots(figsize=(width_fig, height), constrained_layout=True)
    plot_dict = {}
    for i, name in enumerate(model_names):
        plot_dict[name + "_x"] = []
        plot_dict[name + "_y"] = []
        for j in range(i, len(x), len(model_names)):
            plot_dict[name + "_x"].append(x[j])
            plot_dict[name + "_y"].append(y[j])

    x_list = []
    y_list = []
    name_list = []
    for i, name in enumerate(model_names):
        if name == "DELTR n.":
            plt.scatter(plot_dict[name + "_x"], plot_dict[name + "_y"], marker=">", c=colors_points[3], label=name,
                        s=msize, zorder=1)
        elif name == "DELTR":
            plt.scatter(plot_dict[name + "_x"], plot_dict[name + "_y"], marker=">", facecolors='none',
                        edgecolors=colors_points[0], label=name, s=msize, zorder=1)
        elif name == "Clas. n.":
            plt.scatter(plot_dict[name + "_x"], plot_dict[name + "_y"], marker="v", c=colors_points[3], label=name,
                        s=msize, zorder=1)
        elif name == "Clas.":
            plt.scatter(plot_dict[name + "_x"], plot_dict[name + "_y"], marker="v", facecolors='none',
                        edgecolors=colors_points[0], label=name, s=msize, zorder=1)
        elif name == 'FFDR n.':
            if "FFDR" in only_model:
                plt.scatter(plot_dict[name + "_x"], plot_dict[name + "_y"], marker="^", c=colors_points[3], label=name,
                        s=msize, zorder=1)
        elif name == 'FFDR':
            if "FFDR" in only_model:
                plt.scatter(plot_dict[name + "_x"], plot_dict[name + "_y"], marker="^", facecolors='none',
                        edgecolors=colors_points[0], label=name, s=msize, zorder=1)
        elif name == 'ADV FFDR n.':
            if "FFDR" in only_model and "ADV" in only_model:
                plt.scatter(plot_dict[name + "_x"], plot_dict[name + "_y"], marker="s", c=colors_points[3], label=name,
                            s=msize, zorder=1)
        elif name == 'ADV FFDR':
            if "FFDR" in only_model and "ADV" in only_model:
                plt.scatter(plot_dict[name + "_x"], plot_dict[name + "_y"], marker="s", facecolors='none',
                        edgecolors=colors_points[0], label=name, s=msize, zorder=1)
        elif name == "ADV DR n.":
            if "ADV" in only_model:
                plt.scatter(plot_dict[name + "_x"], plot_dict[name + "_y"], marker="o", c=colors_points[3], label=name,
                        s=msize, zorder=1)
        elif name == "ADV DR":
            if "ADV" in only_model:
                plt.scatter(plot_dict[name + "_x"], plot_dict[name + "_y"], marker="o", facecolors='none',
                        edgecolors=colors_points[0], label=name, s=msize, zorder=1)
        elif name == "Con. Opti.":
            plt.scatter(plot_dict[name + "_x"], plot_dict[name + "_y"], marker="x", c=colors_points[0], label=name,
                        s=msize, zorder=1)
        elif name == "Base.":
            plt.scatter(plot_dict[name + "_x"], plot_dict[name + "_y"], marker="x", c=colors_points[1], label=name,
                        s=msize,
                        zorder=1)
            continue
        else:
            continue
        for xi, yi in zip(plot_dict[name + "_x"], plot_dict[name + "_y"]):
            x_list.append(xi)
            y_list.append(yi)
            name_list.append(name)

    without_x_list = []
    without_y_list = []
    with_x_list = []
    with_y_list = []
    for xi, yi, ni in zip(x_list, y_list, name_list):
        print(ni)
        if ni in ["DELTR", "Clas.", "Clas. n.", "Con. Opti."]:
            print(ni)
            without_x_list.append(xi)
            without_y_list.append(yi)
        if "FFDR" in only_model:
            if ni in ['FFDR', 'FFDR n.']:
                with_x_list.append(xi)
                with_y_list.append(yi)
        if "ADV" in only_model:
            if ni in ['ADV DR', 'ADV DR n.']:
                with_x_list.append(xi)
                with_y_list.append(yi)

        if "ADV" in only_model and "FFDR" in only_model:
            if ni in ['ADV DR', 'ADV DR n.', 'ADV FFDR', 'ADV FFDR n.']:
                with_x_list.append(xi)
                with_y_list.append(yi)

    plot_opti(with_x_list, with_y_list, ax, colors[4], ":")
    plot_opti(without_x_list, without_y_list, ax, colors_points[4], "--")

    plt.legend(ncol=3, loc='lower center', frameon=False, prop={'size': 4.5})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title == "law-gender":
        plt.xlim(left=0.6, right=1.05)
        plt.ylim(bottom=0.6, top=1.05)
    elif title == 'adult':
        plt.xlim(left=0.3, right=1.05)
        plt.ylim(bottom=0.3, top=1.05)
    elif title == "law-race":
        plt.xlim(left=0.3, right=1.05)
        plt.ylim(bottom=0.3, top=1.05)
    else:
        plt.xlim(left=0.0, right=1.05)
        plt.ylim(bottom=0.0, top=1.05)
    plt.savefig('newplots/' + title + "_" + eval_ana_name + ylabel + "_2d_plot.pdf")
    plt.savefig('newplots/scatter_png/' + title + "_" + eval_ana_name + ylabel + only_model + "_2d_plot_gpa.pdf")
    plt.close()


def plot_2d_scatter_plots(
        name_list=None,
        x_list=None,
        y_list=None,
        x_label='nDCG@500',
        y_label='1-rND',
        name='ranker',
        data_name=None,
        width_fig=2.35,
        height=2
):
    """
    Plots the 2d scatter plots
    """
    if name == 'repr':
        plot_2d(
            x_list,
            y_list,
            model_names=name_list,
            title=data_name,
            eval_ana_name=name,
            xlabel="nDCG@500",
            ylabel='1-ADRG',
            width_fig=width_fig,
            height=height
        )
    elif name == 'google':
        name_cur = np.array(name_list).tolist()
        name_cur.append("Con. Opti.")
        plot_2d(
            x_list,
            y_list,
            model_names=np.array(name_cur),
            title=data_name,
            eval_ana_name=name,
            xlabel="AUC",
            ylabel='1-GPA',
            width_fig=width_fig,
            height=height
        )
        plot_2d(x_list, y_list, model_names=name_list, title=data_name, eval_ana_name=name, xlabel="AUC",
                ylabel='1-GPA')
    else:
        plot_2d(
            x_list,
            y_list,
            model_names=name_list,
            title=data_name,
            eval_ana_name=name,
            xlabel=x_label,
            ylabel=y_label,
            width_fig=width_fig,
            height=height,
            only_model="FFDR_ADV"
        )


def ndcg_rnd_plot(result_df, filename, width_fig=2.35, height=2, y_value='dr_rnd'):
    """
    Plots the 2d plots for NDCG and RND
    """
    names = [name_2d[name] for name in list(result_df.columns)]
    ndcg = result_df.loc['dr_ndcg']
    rnd = 1 - result_df.loc[y_value]
    if y_value == 'dr_rnd':
        y_label = '1-rND'
    else:
        y_label = '1-GPA'
    plot_2d_scatter_plots(
        name_list=names,
        x_list=ndcg,
        y_list=rnd,
        y_label=y_label,
        name='ranker',
        data_name=filename,
        width_fig=width_fig,
        height=height
    )

def auc_rnd_plot(result_df, filename, width_fig=2.35, height=2, y_value='dr_rnd'):
    """
    Plots the 2d plots for NDCG and RND
    """
    names = [name_2d[name] for name in list(result_df.columns)]
    ndcg = list(result_df.loc['dr_auc'].apply(lambda x: 1 - x if x < 0.5 else x).values)
    rnd = 1 - result_df.loc[y_value]
    if y_value == 'dr_rnd':
        y_label = '1-rND'
    else:
        y_label = '1-GPA'
    plot_2d_scatter_plots(
        name_list=names,
        x_list=ndcg,
        y_list=rnd,
        x_label='AUC',
        y_label=y_label,
        name='ranker_auc',
        data_name=filename,
        width_fig=width_fig,
        height=height
    )


def auc_gpa_plot(result_df, filename, width_fig=2.35, height=2):
    """
    Plots the 2d plots for AUC and GPA
    """
    names = [name_2d[name] for name in list(result_df.columns)]
    # adding results from paper https://arxiv.org/pdf/1906.05330.pdf
    auc = list(result_df.loc['dr_auc'].apply(lambda x: 1 - x if x < 0.5 else x).values)
    gpa = list(1 - result_df.loc['dr_gpa'].values)
    auc.append(0.96)
    gpa.append(1 - 0.01)
    plot_2d_scatter_plots(
        name_list=names,
        x_list=auc,
        y_list=gpa,
        name='google',
        data_name=filename,
        width_fig=width_fig,
        height=height
    )


def representations_plot(result_df, filename, width=0.29, title="", save=True, width_fig=2.35, height=2):
    """
    Plots the bar plots for repr. and returns csv file for the table
    """
    fig, ax = plt.subplots(figsize=(width_fig, height), constrained_layout=True)
    # set fig size to the one in latex
    names = [name_map[name] for name in list(result_df.columns)]
    names_csv = [name_csv[name] for name in list(result_df.columns)]
    rang = np.arange(len(names))
    result_df = result_df.transpose()

    acc_rects = ax.bar(rang - width - 0.01, representation_metric_acc(result_df), width, label='rand. - acc',
                       color=colors[1])
    rnd_rects = ax.bar(rang, representation_metric_rnd(result_df), width, label='1-rND', color=colors[2])
    ndcg_rects = ax.bar(rang + width + 0.01, representation_metric_ndcg(result_df), width, label='nDCG@500',
                        color=colors[3])

    header_csv = []
    names_csv.insert(0, "Models")
    header_csv.extend(names_csv)
    acc = ['ACC']
    rND = ['1-rND']
    ndcg = ['NDCG']
    acc.extend(np.around(representation_metric_acc(result_df).values, 2))
    rND.extend(np.around(representation_metric_rnd(result_df).values, 2))
    ndcg.extend(np.around(representation_metric_ndcg(result_df).values, 2))
    csv_file = np.asarray([header_csv, acc, rND, ndcg])

    for idx, value in enumerate(rang - width):
        plt.text(
            x=value - 0.25,
            y=representation_metric_acc(result_df).values[idx] + 0.01,
            s=round(representation_metric_acc(result_df).values[idx], 2),
            size=3.
        )
    for idx, value in enumerate(rang):
        plt.text(
            x=value - 0.25,
            y=representation_metric_rnd(result_df).values[idx] + 0.01,
            s=round(representation_metric_rnd(result_df).values[idx], 2),
            size=3.
        )
    for idx, value in enumerate(rang + width):
        plt.text(
            x=value - 0.25,
            y=representation_metric_ndcg(result_df).values[idx] + 0.01,
            s=round(representation_metric_ndcg(result_df).values[idx], 2),
            size=3.
        )

    ax.set_xticks(rang)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticks([0, 0.5, 1])
    ax.legend(ncol=3, loc='center', bbox_to_anchor=(0.5, 1.05), frameon=False, prop={'size': 5})
    if save:
        plt.savefig(filename + '.pdf')
    else:
        plt.show()

    return np.transpose(csv_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='I can make plots good')
    parser.add_argument('-p', '--grid_paths', action='store', type=str, nargs='+',
                        help='Path to the directory(ies) containing the gridsearch results.')
    parser.add_argument('-s', '--save', action='store_false', default=True,
                        help='Show, don\'t save')
    args = parser.parse_args()

    metric_name_all_weight = 'weight'
    metric_name_all_weight2 = 'weight2'
    metric_wiki = 'm_wiki'

    df = get_result_df(args.grid_paths)

    df = add_metric_column(df, all_measures_weighted_metric, col_name=metric_name_all_weight2)
    df = add_metric_column(df, gpa_weighted_metric, col_name=metric_name_all_weight)
    df = add_metric_column(df, wiki_metric, col_name=metric_wiki)
    print(df)

    plt.style.use('ecml_paper_figstyle.mplstyle')

    width_fig = 2.35
    height = 2
    results_list = []
    debug_data = True

    plots_path = 'plots'
    os.makedirs(plots_path, exist_ok=True)

    for name in dataset_names:
        if not df[df["dataset"] == name].empty:
            if not df[(df["name"] == "DirectRankerAdv") & (df["gamma"] == 0) & (df["dataset"] == name)].empty:
                with_baseline = True
            else:
                with_baseline = False
            df_all = get_best_all_models(df, name, metric_name_all_weight2, with_baseline=with_baseline,
                                         debug_data=debug_data)
            df_gpa = get_best_all_models(df, name, metric_name_all_weight, with_baseline=with_baseline,
                             debug_data=debug_data)
            if df_all.empty:
                continue
            ndcg_rnd_plot(df_all, name, width_fig=width_fig, height=height)
            ndcg_rnd_plot(df_gpa, name, width_fig=width_fig, height=height, y_value='dr_gpa')
            auc_rnd_plot(df_all, name, width_fig=width_fig, height=height)
            auc_rnd_plot(df_gpa, name, width_fig=width_fig, height=height, y_value='dr_gpa')
            results_list.append(
                representations_plot(df_all, '{}/{}_repr'.format(plots_path, name), title="{} dataset".format(name),
                                     save=args.save, width_fig=width_fig, height=height))
            if name == "wiki":
                df_all = get_best_all_models(df, name, metric_wiki, with_baseline=with_baseline, debug_data=debug_data)
            auc_gpa_plot(df_all, name, width_fig=width_fig, height=height)

        else:
            print("No dataset " + name)

    final_table = get_repr_csv(results_list)
