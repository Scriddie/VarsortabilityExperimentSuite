import os
import shutil
import numpy as np
import itertools
import pickle as pk
from hashlib import md5
import sys
print(sys.path[0])
sys.path.append(sys.path[0]+'/..')
import _Scalers as Scalers
import _utils as utils
import sys
np.set_printoptions(suppress=False)
np.set_printoptions(threshold=sys.maxsize)  # don't truncate big arrays


class DataGenerator:
    def __init__(self, exp_name, base_dir, scaler, random_seed=True):
        """ 
        Generate and fill _data dir in exp_name parent folder. Overwrites old data.
        Args:
            exp_name(String): Name of the experiment
            base_dir(String): Dir for output folders
            scaler(Scaler): Type of scaling
        """
        
        self.scaler = scaler if scaler else Scalers.Identity()
        self.random_seed = random_seed
        # output dir
        utils.create_folder(base_dir)
        utils.create_folder(os.path.join(base_dir, exp_name))
        # output folder
        output_folder = os.path.join(base_dir, exp_name, "_data")
        utils.overwrite_folder(output_folder)
        self.output_folder = output_folder

    def create_data(self, description, W_true):
        """ Create sample for dataset """
        if self.random_seed:
            utils.set_random_seed(description.random_seed)
        if description.noise_variance:
            l, u = description.noise_variance
            noise_scales = np.random.uniform(l, u, size=description.n_nodes)
        else:
            noise_scales = None
        B_true = np.where(W_true!=0, 1, 0)
        if description.noise in ['gp-add', 'mim', 'mlp', 'gp', 'gp-add-lach']:
            data = utils.simulate_nonlinear_sem(B_true,
                                                description.n_obs,
                                                sem_type=description.noise,
                                                noise_scale=noise_scales)
        else:
            data = utils.simulate_linear_sem(W_true, 
                                             description.n_obs,
                                             sem_type=description.noise,
                                             noise_scale=noise_scales)
        data = self.scaler.transform(data)
        varsortability = utils.varsortability(data, W_true, False)
        return utils.Dataset(description=description,
                             W_true=W_true,
                             B_true=B_true,
                             data=data,
                             hash=md5(data).hexdigest()[0:10],
                             scaler=self.scaler.name(),
                             scaling_factors=self.scaler.scaling_factors,
                             varsortability=varsortability)

    def create_dataset(self, description):
        """ 
        Create a single Dataset according to specifications
        Args:
            description(Dataset): Namedtuple of Dataset descriptions
        """
        if self.random_seed:
            utils.set_random_seed(description.random_seed)
        graph = description.graph.split("-")[0]
        expected_edges = int(description.graph.split("-")[1]) * int(description.n_nodes)
        B_true = utils.simulate_dag(description.n_nodes, expected_edges, graph)
        w = description.edge_weight_range
        if isinstance(w, int):
            w_ranges = ((w, w), (w, w))
        else:
            w_ranges = (tuple([-i for i in reversed(list(description.edge_weight_range))]),
                description.edge_weight_range)
        W_true = utils.simulate_parameter(B_true, w_ranges=w_ranges)
        return self.create_data(description, W_true)


    def get_descriptions(self,
                         graphs, 
                         noise_distributions,
                         edge_weights,
                         n_nodes,
                         n_obs,
                         n_repetitions):
        """ Generate Dataset descriptions as permutations of inputs 
        Args:
            graphs(list): ["ER-2", "SF-4"]
            noise_distributions(list): Noise distribution with variance (c.f. utils)
            edge_weights(list): Intervals
            n_nodes(list): Number of nodes in graph
            n_obs(list): Number of observations in sample
        """
        combinations = list(itertools.product(graphs,
                                              noise_distributions,
                                              edge_weights,
                                              n_nodes,
                                              n_obs,
                                              list(range(n_repetitions))))
        return [utils.DatasetDescription(
                graph=gt,
                noise=nd.type,
                noise_variance=nd.uniform_variance,
                edge_weight_range=ew,
                n_nodes=nn,
                n_obs=no,
                random_seed=rs) for (gt, nd, ew, nn, no, rs) in combinations]

    def generate(self,
                 n_repetitions,
                 graphs, 
                 noise_distributions,
                 edge_weights,
                 n_nodes,
                 n_obs):
        """ Generate Data
        Args:
            All params are lists
        Returns:
            All Datasets
        """
        dataset_descriptions = self.get_descriptions(graphs, 
                                                     noise_distributions,
                                                     edge_weights,
                                                     n_nodes,
                                                     n_obs,
                                                     n_repetitions)
        return [self.create_dataset(i) for i in dataset_descriptions]

    def generate_and_save(self,
                          n_repetitions,
                          graphs, 
                          noise_distributions,
                          edge_weights,
                          n_nodes,
                          n_obs):
        """ Generate and save Dataset data """

        # create Datasets
        all_datasets = self.generate(n_repetitions,
                                     graphs,
                                     noise_distributions,
                                     edge_weights,
                                     n_nodes,
                                     n_obs)

        # create parent directories
        unique_dirs = set([os.path.join(self.output_folder, utils.dataset_dirname(dataset))
                           for dataset in all_datasets])
        for d in unique_dirs:
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.mkdir(d)

        # save datasets
        checksums = [("name,checksum")]
        prev_dir = ""
        for exp_idx, dataset in enumerate(all_datasets):
            dirname =  utils.dataset_dirname(dataset)
            fname = utils.dataset_description(dataset)
            dir = os.path.join(self.output_folder, dirname)
            
            hashsum = md5(dataset.data).digest()
            checksums.append(f"{fname},{hashsum}")

            fdir = os.path.join(dir, fname) + ".pk"
            with open(fdir, 'wb') as f:
                pk.dump(dataset, f)

            if dirname != prev_dir:
                print(exp_idx, "COMPLETED", dirname)
                prev_dir = dirname

        with open(os.path.join(self.output_folder, "checksums.csv"), "w") as f:
            f.write("\n".join(checksums))


if __name__ == "__main__":
    """ Run DataGenerator from command line """
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate data")

    # experiment specifications
    parser.add_argument("--exp_base_name", type=str, help="experiment main folder", required=True)
    parser.add_argument("--exp_name", type=str, help="experiment name", required=True)
    parser.add_argument("--base_dir", type=str, help="data output dir", default="src/experiments")
    parser.add_argument("--scaler", type=str, help="e.g. 'identity'", required=True)

    # experiment settings
    parser.add_argument("--n_repetitions", type=int, help="number of repetitions per combination", default=10)
    parser.add_argument("--graph_types", type=str, nargs="+", help="e.g. ER-2", default=["ER-2"])
    parser.add_argument("--n_nodes", type=int, nargs="+", help="e.g. 5 10 for graphs with 5 and 10 nodes", default=[5, 10])
    parser.add_argument("--n_obs", type=int, help="number of observations per sample", default=1000)

    args = parser.parse_args()

    if args.scaler == "identity":
        scaler = Scalers.Identity()
    elif args.scaler == "normalizer":
        scaler = Scalers.Normalizer()
    else:
        raise ValueError("No such scaler?")

    """ Data Generation """
    settings = {
        "n_repetitions": args.n_repetitions,
        "graphs": args.graph_types,
        "noise_distributions": [
            # utils.NoiseDistribution("gauss", (1, 1)),
            utils.NoiseDistribution("gauss", (0.5, 2)),
            # utils.NoiseDistribution("exp", (0.5, 2)),
            # utils.NoiseDistribution("gumbel", (0.5, 2)),
        ],
        "edge_weights":[(.5, 2)],
        "n_nodes": args.n_nodes,
        "n_obs": [args.n_obs],
    }
    dg = DataGenerator(args.exp_name, os.path.join(args.base_dir, args.exp_base_name), scaler=scaler)
    dg.generate_and_save(**settings)