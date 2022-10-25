### Run piecewise for easier parallelization. In case of manual parallelization with bash scripts, using ray is not necessary.
### To be run from repository root directory

# set-up
mkdir -p src/experiments
source env/bin/activate
exp_base_name="default_piecewise"
exp_raw="${exp_base_name}_raw"
exp_normalized="${exp_base_name}_normalized"
n_nodes=5
rm -rf "src/experiments/${exp_base_name}"

# generate data
python src/_DataGenerator.py --exp_base_name $exp_base_name \
    --exp_name $exp_raw --scaler "identity" --n_repetitions 10 \
    --graph_types "ER-2" --n_nodes $n_nodes --n_obs 1000
python src/_DataGenerator.py --exp_base_name $exp_base_name \
    --exp_name $exp_normalized --scaler "normalizer" --n_repetitions 10 \
    --graph_types "ER-2" --n_nodes $n_nodes --n_obs 1000

# run 'sortnregressIC' on generated data
python src/_ExperimentRunner.py --exp_base_name $exp_base_name \
    --exp_name $exp_raw --algorithm 'sortnregressIC' --no_ray
python src/_ExperimentRunner.py --exp_base_name $exp_base_name \
    --exp_name $exp_normalized --algorithm 'sortnregressIC' --no_ray
# run 'notearsLinear' on generated data
python src/_ExperimentRunner.py --exp_base_name $exp_base_name \
    --exp_name $exp_raw --algorithm 'notearsLinear' --no_ray
python src/_ExperimentRunner.py --exp_base_name $exp_base_name \
    --exp_name $exp_normalized --algorithm 'notearsLinear' --no_ray

# evaluation
python src/_Evaluator.py --exp_base_name $exp_base_name \
    --exp_name $exp_raw --threshold "0.3" --type "standard"
python src/_Evaluator.py --exp_base_name $exp_base_name \
    --exp_name $exp_normalized --threshold "0.3" --type "standard"

# visualization
python src/_Visualizer.py --exp_base_name $exp_base_name --exp_name $exp_raw \
    $exp_normalized --n_nodes $n_nodes --threshold 0.3 \
    --thres_type standard --overwrite_prev