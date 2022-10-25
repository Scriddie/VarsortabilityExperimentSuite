"""
Visualize a result file
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import _utils as utils
import sys
from operator import itemgetter
import re
import warnings
np.set_printoptions(suppress=False)
np.set_printoptions(threshold=sys.maxsize)  # don"t truncate big arrays
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 18


class Visualizer:
    def __init__(self, exp_name, base_dir, overwrite_prev, n_nodes, threshold=0.001, thres_type="standard"):
        """
        Load a result df from path
        Args:
            exp_name(String): Name of the experiment
            base_dir(String): parent dir of data and viz
            overwrite_prev(bool): Overwrite _viz folder
            thres_type: Which thresholding type to use
        """
        self.exp_names = exp_name if isinstance(exp_name, list) else [exp_name]
        self.threshold = str(threshold)
        self.n_nodes = n_nodes

        # read from all experiments provided
        self.res_dfs, self.res_df_names = [], []
        for i in self.exp_names:
            input_dir = os.path.join(base_dir, i, "_eval")
            results_files = list(Path(input_dir).rglob(f"{thres_type}_{threshold}.csv"))
            for f in results_files:
                print("Reading from", f, "...")
                df = utils.load_results(f)
                df["experiment"] = i
                name = f.stem
                # merge datasets from different experiments for comparison
                if name in self.res_df_names:
                    idx = self.res_df_names.index(name)
                    self.res_dfs[idx] = pd.concat((self.res_dfs[idx], df), axis=0)
                    print(f"...merging across experiments into {name}")
                else:
                    self.res_dfs.append(df)
                    self.res_df_names.append(name)

        # output
        self.exp_name = "-vs-".join(self.exp_names)
        self.output_dir = os.path.join(base_dir, self.exp_name)
        utils.create_folder(self.output_dir)
        head_folder = os.path.join(self.output_dir, "_viz")
        utils.create_folder(head_folder)
        self.output_folder = os.path.join(head_folder, f"thr_{self.threshold}_{self.n_nodes}nodes")
        if overwrite_prev:
            utils.overwrite_folder(self.output_folder)
        else:
            utils.create_folder(self.output_folder)


    def _decorator(func):
        """
        Common decorator for all plotting functions
        Args:
            acc_measure(String): Evaluation metric to plot
            filters(dict): Filter arguments, e.g. "graph: "ER-2""
            algo_list(list): List of names of algorithms to include
            display(bool): Whether or not to show plot immediately
        """
        def function_wrapper(self, acc_measure, filters={}, *args, **kwargs):
            plt.close("all")
            function_name = func.__name__

            # preprocessing
            for i, res_df in enumerate(self.res_dfs):
                def _select_data(filters, df):  
                    for k, v in filters.items():
                        v_filter = [v] if not isinstance(v, list) else v
                        mask = df[k].apply(lambda x: x in v_filter)
                        df = df.loc[mask, :]
                    return df
                subset = _select_data(filters, res_df.copy())

                # check if empty
                if len(subset) == 0:
                        continue

                # ordering
                if "algorithm" in filters.keys():
                    subset["algorithm"] = subset["algorithm"].apply(lambda x: str(filters["algorithm"].index(x))+x)
                    subset.sort_values(by=["algorithm", "scaler"], inplace=True)
                    subset["algorithm"] = subset["algorithm"].apply(lambda x: re.sub("[0-9]", "", x))

                if subset[acc_measure].min() < 0:
                    message = f"min {acc_measure.upper()} < 0. Likely cause: {acc_measure.upper()} evaluation not implemented/active."
                    warnings.warn(message)
                    continue
                g = func(self, subset, res_df, acc_measure, *args)
                
                # save plot
                name_list = list(kwargs.keys()) + list(filters.keys())
                g.savefig(f"{self.output_folder}/{function_name}_{self.res_df_names[i]}-{('_').join(name_list)}_{acc_measure}.pdf".replace(" ", ""), dpi=200)
                plt.close("all")
        return function_wrapper


    @_decorator
    def boxplot(self, dataset, raw_res_df, acc_measure, static_xlim=False):
        if static_xlim: 
            upper_xlim = np.max(raw_res_df[self.ugly_axes_names[acc_measure]])
        else: 
            upper_xlim = np.max(dataset[acc_measure])
        fig, ax = plt.subplots(figsize=(10, 5.5))  # (10, 5)
        ax.set_facecolor("#FFFFFF")
        plt.grid(True, linewidth=0.5, color="#999999", linestyle="-")
        ax.set_axisbelow(True)
        sns.boxplot(x=dataset[acc_measure],y=dataset["algorithm"], hue=dataset["experiment"].apply(lambda x: x.split("_")[-1]), showfliers=False, ax=ax, palette=itemgetter(7, 3, 1, 2, 4)(sns.color_palette("Paired")), zorder=1)
        sns.despine(left=True, bottom=True)
        ax.set_xlim(0, upper_xlim)
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Visualize results")

    # experiment specifications
    parser.add_argument("--exp_base_name", type=str, help="experiment main folder", required=True)
    parser.add_argument("--exp_name", type=str, nargs="+", help="experiment names", required=True)
    parser.add_argument("--n_nodes", type=int, help="visualize boxplots for this number of nodes", required=True)
    parser.add_argument("--base_dir", type=str, help="data input dir", default="src/experiments")
    parser.add_argument("--overwrite_prev", action="store_true", default=False, help="overwrite _viz directory")
    parser.add_argument("--threshold", type=str, help="threshold for thresholding algos", default="0.3")
    parser.add_argument("--thres_type", type=str, default="standard")
    parser.add_argument("--static_xlim", action="store_true", default=False)

    args = parser.parse_args()
    viz = Visualizer(args.exp_name, os.path.join(args.base_dir, args.exp_base_name), overwrite_prev=args.overwrite_prev, n_nodes=args.n_nodes, threshold=float(args.threshold), thres_type=args.thres_type)


    if len(args.exp_name) == 1:  # single one of raw/normalized
        viz.facet_wrap(acc_measure="sid")
        viz.facet_wrap(acc_measure="shd")
        viz.facet_wrap(acc_measure="runtime")
        for var in ["(0.5, 2)"]:
            filters={"noise_variance": var}
            for acc_measure in ["shd"]:
                for noise in [["gauss"]]:
                    for graph in ["ER-2"]:
                        viz.boxplot(acc_measure=acc_measure, filters={"n_nodes": args.n_nodes, "noise": noise, "noise_variance": var, "graph": graph}, custom_name="selection", static_xlim=args.static_xlim)

    elif len(args.exp_name) == 2:  # comparison raw vs. normalized
        for var in ["(1, 1)", "(0.5, 2)"]:
            filters={"noise_variance": var}
            for acc_measure in ["shd"]:  # ["sid", "shd"]
                for noise in [["gauss"], ["exp"], ["gumbel"]]:
                    for graph in ["ER-1", "ER-2", "ER-4", "SF-4"]:
                        viz.boxplot(acc_measure=acc_measure, filters={"n_nodes": args.n_nodes, "noise": noise, "noise_variance": var, "graph": graph}, static_xlim=args.static_xlim)
    else:
        print("Unknown number of arguments to Visualizer")
