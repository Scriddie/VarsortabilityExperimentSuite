""" Basic experiment setup of 'Beware of the simulated DAG' """
import sys
sys.path.append(sys.path[0]+'/src')
sys.path.append(sys.path[0]+'/golem/src')
import os
import ray
from _DataGenerator import DataGenerator
from _ExperimentRunner import ExperimentRunner
from _Evaluator import Evaluator
from _Visualizer import Visualizer
import _Scalers as Scalers
import _utils as utils


def data(params, scaler, data_params):
    """ Data Generation """
    dg = DataGenerator(params['version_name'], params['base_dir'], scaler=scaler)
    dg.generate_and_save(**data_params)
 
 
def experiment(params):
    """ Run a sel of algorithms on the experiment data """
    max_mem = 4 * 1024 * 1024 * 1024  # 4GB
    expR = ExperimentRunner(params['version_name'], 
                            params['base_dir'], 
                            overwrite_prev=False,       
                            num_cpus=5, 
                            _redis_max_memory=max_mem, 
                            use_ray=False)
    for i in params['algorithms']:
        eval('expR.'+i+'()')


def evaluate(params):
    """ Evaluation """
    ev = Evaluator(params['version_name'], params['base_dir'], overwrite_prev=False, threshold=params['thres'], MEC=params['MEC'])
    ev.evaluate(thresholding=params['thres_type'])
  
 
def visualize(params, data_params):
    """ Visualization """
    n_nodes = max(data_params['n_nodes'])
    viz = Visualizer(params['version_name'], params['base_dir'], overwrite_prev=False, thres_type=params['thres_type'], n_nodes=n_nodes, threshold=params['thres'])
    acc_measures = ["sid", "shd"]
    if params['MEC']: acc_measures += ["mec_sid", "mec_shd"]
    for noise in [["gauss"], ["exp"], ["uniform"]]:
        for acc_measure in acc_measures:
            viz.boxplot(acc_measure=acc_measure, filters={
                "n_nodes": n_nodes, 
                "noise": noise, 
                "noise_variance": "(0.5, 2)",
                "graph": "ER-2"})

 
def viz_compare(params, data_params, name1, name2):
    """ do a x vs. y evaluation """
    n_nodes = max(data_params['n_nodes'])
    viz = Visualizer([
        params['name']+name1, params['name']+name2], 
        params['base_dir'],
        overwrite_prev=False, 
        thres_type=params['thres_type'], 
        n_nodes=n_nodes, 
        threshold=params['thres']
    )
    acc_measures = ["shd"]  # ["sid", "shd"]
    if params['MEC']: acc_measures += ["mec_sid_upper", "mec_shd"]
    for acc_measure in acc_measures:
        for noise in [["gauss"], ["exp"], ["gumbel"], ["uniform"]]:
            for var in ["(1, 1)", "(0.5, 2)", "(2, 3)"]:
                for graph in ["ER-2", "ER-4", "SF-4"]:
                    viz.boxplot(acc_measure=acc_measure, filters={
                        "n_nodes": n_nodes, 
                        "noise": noise, 
                        "noise_variance": var, 
                        "graph": graph})
 
 
def run(params, version, scaler, data_params):
   """ Complete experiment """
   params['version_name'] = params['name'] + version
   data(params, scaler, data_params)
   experiment(params)
   evaluate(params)
   visualize(params, data_params)


if __name__ == "__main__":
    # data generation
    data_params = {
        "n_repetitions": 10,
        "graphs": ["ER-2"],
        "noise_distributions": [utils.NoiseDistribution("gauss", (0.5, 2))],
        "edge_weights":[(0.5, 2)],
        "n_nodes": [5],
        "n_obs": [1000],  # for less than ~1000, the tetrad algos fail
    }

    # evaluation parameters
    params = {
        'MEC': False,
        'exp_dir': os.path.join("src", "experiments"),
        'name': "default",
        'thres': 0.3,
        'thres_type': "standard",
        # algorithms from _ExperimentRunner as strings
        'algorithms': [
            # 'golemEV_golemNV_orig',
            'notearsLinear',
            'sortnregressIC',
        ]
    }
    params['base_dir'] = os.path.join(params['exp_dir'], params['name'])

    # create folders
    utils.set_random_seed(0)
    utils.create_folder(params['exp_dir'])
    utils.create_folder(params['base_dir'])

    # run experiment
    run(params.copy(), "_raw", Scalers.Identity(), data_params)
    run(params, "_normalized", Scalers.Normalizer(), data_params)
    viz_compare(params, data_params, "_raw", "_normalized")

    # kill any open processes
    ray.shutdown()
