# Run VarsortabilityExperimentSuite
1. Install the requirements and dependencies (script and more info: **setup.sh**)
2. Run `src/run_experiment.py` from repo root directory with active environment (script and more info: **run.sh**, **run_piecewise.sh**)
3. Results can be found at `src/experiments/default/default_raw-vs-default_normalized`

# Analysis Pipeline
1. **_utils** Bastic metrics and data generation utilities
2. **_DataGenerator** Generates a lot of data systematically
3. **_Scaler** Scales data (e.g. standardize)
4. **_ExperimentRunner** Run algos and save results
5. **_Evaluator** Creates evaluations (SHD/SID/...) from algo results
6. **_Visualizer** Creates plots from evaluations
- **run_experiment.py** Specify experiment and run analysis pipeline

# Versions
Tested with Python 3.6.9.