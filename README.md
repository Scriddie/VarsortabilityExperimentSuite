# Content
This repository comprises the **basic experimental set-up for the comparison of causal structure learning algorithms** as shown in

[1] Reisach, A. G., Seiler, C., & Weichwald, S. (2021). [Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game](https://proceedings.neurips.cc/paper/2021/file/e987eff4a7c7b7e580d659feb6f60c1a-Paper.pdf).

For stand-alone implementations of **varsortability**, **sortnregress**, and **chain-orientation** as presented in the same work, see the [Varsortability](https://github.com/Scriddie/Varsortability) repository.

If you find this code useful, please consider citing:
```
@article{reisach2021beware,
  title={Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy to Game},
  author={Reisach, Alexander G. and Seiler, Christof and Weichwald, Sebastian},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

## Run VarsortabilityExperimentSuite
1. Install the requirements and dependencies (script and more info: **`setup.sh`**)
2. Run `src/run_experiment.py` from repo root directory with active environment (script and more info: **`run.sh`**, **`run_piecewise.sh`**)
3. Results can be found at `src/experiments/default/default_raw-vs-default_normalized`

## Analysis Pipeline
1. **_utils** Bastic metrics and data generation utilities
2. **_DataGenerator** Generates a lot of data systematically
3. **_Scaler** Scales data (e.g. standardize)
4. **_ExperimentRunner** Run algos and save results
5. **_Evaluator** Creates evaluations (SHD/SID/...) from algo results
6. **_Visualizer** Creates plots from evaluations
- **run_experiment.py** Specify experiment and run analysis pipeline

## Versions
Tested with Python 3.6.9.