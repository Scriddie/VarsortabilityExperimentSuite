import os
import sys
import numpy as np
import pandas as pd
import pickle as pk
from datetime import datetime
from tabulate import tabulate
import git
import ray
import gc
import time
import _utils as utils
from pathlib import Path
sys.path.append(sys.path[0]+'/src/golem/src')
np.set_printoptions(suppress=False)
np.set_printoptions(threshold=sys.maxsize)  # don't truncate big arrays


class ExperimentRunner:
    def __init__(self,
                 exp_name,
                 base_dir,
                 overwrite_prev,
                 ray_address="auto",
                 redis_pw="jtbs",
                 num_cpus=None,
                 num_gpus=0,
                 _redis_max_memory=None,
                 use_ray=True):
        """
        Run algorithms on datasets, save data in output_dir
        Args:
            datasets(list): datasets (c.f. utils)
            base_dir(String): load datasets from here!
            overwrite_prev(String): Overwrite "_res" folder
        """
        assert (num_cpus and _redis_max_memory) or (not num_cpus and  not _redis_max_memory)

        # parent dir of all the experiments
        self.input_folder = os.path.join(base_dir, exp_name, "_data")
        self.output_dir = os.path.join(base_dir, exp_name)
        utils.create_folder(base_dir)
        utils.create_folder(self.output_dir)
        self.output_folder = os.path.join(self.output_dir, "_res")
        if overwrite_prev: utils.overwrite_folder(self.output_folder)
        else: utils.create_folder(self.output_folder)
        self.overwrite_prev = overwrite_prev

        # gather all data files, only load when necessary
        self.data_files = [i for i in list(Path(self.input_folder).rglob("*.pk"))]
        self.git_hash = git.Repo(search_parent_directories=True).head.object.hexsha[0:10]
        self.num_gpus = num_gpus

        self.use_ray = use_ray
        if not ray.is_initialized() and use_ray:
            try:
                ray.init(address=ray_address, _redis_password=redis_pw)
            except ConnectionError:
                if num_cpus and _redis_max_memory:
                    ray.init(num_cpus=num_cpus, _redis_max_memory=_redis_max_memory, _redis_password="jtbs")
                else:
                    ray.init(_redis_password="jtbs")
                print("New raylet created;")

    @ray.remote
    def _log_results_remote(self, func, data_file, *args, **kwargs):
        self._log_results(func, data_file, *args, **kwargs)

    def _log_results_local(self, func, data_file, *args, **kwargs):
        self._log_results(func, data_file, *args, **kwargs)

    def _log_results(self, func, data_file, *args, **kwargs):
        """ 
        For each single dataset: Common functionality for all DAG-fitting methods:
            1. thresholding
            2. parallelization
            3. logging
        """
        dataset = pk.load(open(data_file, "rb"))
        algo_name = func.__name__
        if dataset.description.random_seed ==0:
            print(algo_name.upper(), dataset.description)

        # actual function
        start_time = str(datetime.now())
        start = time.time()
        W_est = func(self, dataset, *args, **kwargs)
        end = time.time()
        runtime = end - start
        W_est = W_est.astype("float64")

        # create results folder
        exp_combo_name = data_file.parent.name
        exp_combo_folder = os.path.join(self.output_folder, exp_combo_name)
        utils.create_folder(exp_combo_folder)
        results = {
            "algorithm":            [algo_name],
            "hyperparams":          [kwargs],
            "scaler":               [dataset.scaler],
            "scaling_factors":      [[dataset.scaling_factors]],
            "dataset_description":  [utils.dataset_description(dataset)],
            "dataset_hash":         [dataset.hash],
            "noise_variance":       [],
            "edge_weight_range":    [],
            "n_nodes":              [],
            "n_obs":                [],
            "random_seed":          [],
            "graph":                [],
            "noise":                [],
            "start_time":           [start_time],
            "runtime":              [np.round(runtime, 0)],
            "git_hash":             [self.git_hash],
            "W_true":               [[np.copy(dataset.W_true)]],
            "W_est":                [[np.copy(W_est)]],
            "varsortability":       [dataset.varsortability],
        }
        for k, v in dataset.description._asdict().items():
            results[k].append(v)

        # write file
        fname = f"{func.__name__}_{data_file.stem}_results.csv"
        fpath = os.path.join(exp_combo_folder, fname)
        self.collect_results(results, fpath, display=dataset.description.random_seed==0)

        del dataset
        gc.collect()

    def _algo_decorator(func):
        """ Apply each DAG estimation function to all .pk files in base_dir, log and save result. """
        def wrapper(self, *args, **kwargs):
            if self.use_ray:
                res_ids = []
                for data_file in self.data_files:
                    res_ids.append(self._log_results_remote.options(num_gpus=self.num_gpus).remote(self, func, data_file, *args, **kwargs))
                _ = ray.get(res_ids)
            else:
                for data_file in self.data_files:
                    # print("file:", data_file)
                    self._log_results_local(func, data_file, *args, **kwargs)
        return wrapper

    @_algo_decorator
    def notearsLinear(self, dataset, lambda1=0., loss_type="l2"):
        from notears.notears.linear import notears_linear
        return notears_linear(dataset.data, lambda1, loss_type)

    @_algo_decorator
    def golemEV_orig(self, dataset, episodes=int(1e+4), de_mean=True, save_learning_curves=True):
        from golem.src.golem import golem
        X = dataset.data - np.mean(dataset.data, axis=0) if de_mean else dataset.data
        return golem(X, lambda_1=2e-2, lambda_2=5.0, equal_variances=True,
            num_iter=episodes, learning_rate=1e-3, seed=np.random.randint(0, 2**32-1), B_init=None, checkpoint_iter=None, output_dir=None)

    @_algo_decorator
    def golemNV_orig(self, dataset, episodes=int(1e+4), de_mean=True):
        from golem.src.golem import golem
        X = dataset.data - np.mean(dataset.data, axis=0) if de_mean else dataset.data
        return golem(X, lambda_1=2e-3, lambda_2=5-0, equal_variances=False,
            num_iter=episodes, learning_rate=1e-3, seed=np.random.randint(0, 2**32-1), B_init=None, checkpoint_iter=None, output_dir=None)

    @_algo_decorator
    def golemEV_golemNV_orig(self, dataset, episodes=int(1e+4)):
        from golem import golem
        B_init = golem(np.copy(dataset.data), lambda_1=2e-2, lambda_2=5.0,
            equal_variances=True, num_iter=episodes, learning_rate=1e-3, seed=np.random.randint(0, 2**32-1), B_init=None, checkpoint_iter=None, output_dir=None)
        return golem(dataset.data, lambda_1=2e-3, lambda_2=5.0,
            equal_variances=False, num_iter=episodes, learning_rate=1e-3, seed=np.random.randint(0, 2**32-1), B_init=B_init, checkpoint_iter=None, output_dir=None)

    @_algo_decorator
    def notearsNonlinear(self, dataset, lambda1, lambda2):
        from notears.nonlinear import notears_nonlinear, NotearsMLP
        model = NotearsMLP(dims=[dataset.description.n_nodes, 10, 1], bias=True)
        return notears_nonlinear(model, dataset.data, lambda1=lambda1,  
            lambda2=lambda2)

    @_algo_decorator
    def sortnregressIC(self, dataset):
        from sortnregress import sortnregress
        return sortnregress(dataset.data, regularisation="bic", random_order=False)

    @_algo_decorator
    def empty(self, dataset):
        d = dataset.description.n_nodes
        return np.zeros((d, d))

    def collect_results(self, res_dict, path, save=True, display=False):
        """ Save and print result dataframe """
        res_df = pd.DataFrame(res_dict)
        res_df.sort_values(by=["algorithm", "start_time"], ascending=True, inplace=True)
        res_df.drop_duplicates(subset=["algorithm", "dataset_hash"], keep="last", inplace=True)
        if save:
            with np.printoptions(threshold=sys.maxsize, suppress=False):
                res_df.to_csv(path, index=False)
        if display:
            display_cols = ["algorithm", "dataset_description", "start_time", "runtime"]
            print(tabulate(res_df[display_cols], headers='keys', tablefmt='psql'))


if __name__ == "__main__":
    """ Run ExperimentRunner from command line """
    import argparse
    import ray
    import os
    
    parser = argparse.ArgumentParser(description="Run an experiment")

    # experiment specifications
    parser.add_argument("--exp_base_name", type=str, help="experiment main folder", required=True)
    parser.add_argument("--exp_name", type=str, help="experiment name", required=True)
    parser.add_argument("--base_dir", type=str, help="data input dir", default="src/experiments/")
    parser.add_argument("--algorithm", type=str, help="algorithm name", required=True)
    parser.add_argument("--overwrite_prev", action="store_true", default=False, help="Overwrite _res directory")
    parser.add_argument("--ray_address", type=str, help="number of cpus", default="auto")
    parser.add_argument("--redis_pw", type=str, help="number of cpus", default="jtbs")
    parser.add_argument("--no_ray", action="store_false", help="do not use ray parallelization")

    args = parser.parse_args()
    expR = ExperimentRunner(
        args.exp_name, 
        os.path.join(args.base_dir, args.exp_base_name),
        args.overwrite_prev, args.ray_address, args.redis_pw,
        num_cpus=1, _redis_max_memory=4 * 1024 * 1024 * 1024,
        use_ray=args.no_ray)

    eval('expR.'+args.algorithm+'()')

    # utils.stop_pycausalvm()
    ray.shutdown()