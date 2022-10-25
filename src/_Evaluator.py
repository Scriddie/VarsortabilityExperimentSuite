""" Central evaluation logic """
import _utils as utils
import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
np.set_printoptions(suppress=False)  # scientific notation is ok
np.set_printoptions(threshold=sys.maxsize)  # don't truncate big arrays
R_LIBS = False
if R_LIBS:
    import rpy2.robjects.numpy2ri as rn
    import rpy2.robjects as ro
    rn.activate()
    ro.r("""library(SID)""")


class Evaluator:
    def __init__(self,
                 exp_name,
                 base_dir,
                 overwrite_prev,
                 threshold=0.001,
                 MEC=False):
        """
        Evaluate results in _res of exp_name
        Args:
            exp_name(String): Name of the experiment
            base_dir(String): Load datasets from here
            overwrite_prev(String): Remove all previous evaluations
        """

        # parent dir of all the experiments
        self.input_folder = os.path.join(base_dir, exp_name, "_res")
        output_dir = os.path.join(base_dir, exp_name)
        utils.create_folder(base_dir)
        utils.create_folder(output_dir)
        self.output_folder = os.path.join(output_dir, "_eval")
        if overwrite_prev:
            utils.overwrite_folder(self.output_folder)
        else:
            utils.create_folder(self.output_folder)
        self.threshold = threshold
        self.MEC = MEC

    def read_files(self):
        # gather all .csv files recursevely
        results_files = list(Path(self.input_folder).rglob("*results.csv"))
        all_df = []
        names = []
        for file in results_files:
            if "varsortability" in file.name:
                continue
            all_df.append(utils.load_results(file))
            names.append(file.name)
        return pd.concat(all_df, axis=0).to_dict("list"), names

    def evaluate(self, thresholding="standard", eligible=[]):
        """ 
        Evaluate a given solution in results
        Args:
            thresholding(String): One of ["standard", "dynamic", "favourable"]
            eligible(list): List of algorithms eligible for non-standard thresholding
        """
        inputs, names = self.read_files()
        keys = list(inputs.keys()) + [ "fdr", "tpr", "fpr", "nnz", "sid", "shd", "mec_sid", "mec_sid_lower", "mec_sid_upper", "mec_shd", "was_dag", "effective_thresholding", "w_threshold"]
        results = {i: [] for i in keys}
        for i in range(len(inputs["W_est"])):
            W_true = inputs["W_true"][i][0]
            B_true = W_true != 0
            W_est = inputs["W_est"][i][0]
            # copy all the previous entries
            for k, v in inputs.items():
                results[k].append(v[i])
            
            W_est_thr = np.copy(W_est)
            was_dag = utils.is_dag(W_est_thr)

            # thresholding
            w_threshold_mapping = utils.thresholds(self.threshold)
            algo = inputs["algorithm"][i]
            assert algo in w_threshold_mapping, f"No threshold for {algo}"
            w_threshold = w_threshold_mapping[algo]

            if thresholding == "standard": 
                effective_thresholding = "standard"
            elif thresholding == "favourable":
                if algo in eligible:
                    effective_thresholding = "favourable"
                    w_threshold = np.mean(np.sort(np.abs(W_est_thr).ravel())[::-1][(B_true != 0).sum()-1:(B_true != 0).sum()+1])
                else:
                    effective_thresholding = "standard"
            else:
                raise ValueError("no such thresholding")

            # threshold
            if w_threshold == 'bidirected':
                # we encode negative numbers for the wrong direction in CPDAGs
                W_est_thr[W_est_thr < 0.] = 0
            else:
                W_est_thr[np.abs(W_est_thr) < w_threshold] = 0

            # turn into DAG
            while not utils.is_dag(W_est_thr):
                W_est_thr = utils.dagify(W_est_thr)

            B_est = W_est_thr != 0

            # Accuracy measures
            acc = utils.count_accuracy(B_true, B_est)
            for k, v in acc.items():
                results[k].append(v)
            
            # SID
            if R_LIBS:
                nr, nc = B_true.shape
                r_B_true = ro.r.matrix(B_true, nrow=nr, ncol=nc)
                r_B_est = ro.r.matrix(B_est, nrow=nr, ncol=nc)
                ro.r("""sid_fun <- function(b_true, b_est){structIntervDist(b_true, b_est)$sid}""")
                r_sid_fun = ro.globalenv["sid_fun"]
                sid = r_sid_fun(r_B_true, r_B_est)[0]
                results["sid"].append(sid)
            else:
                results["sid"].append(-1)

            if self.MEC:
                assert R_LIBS, 'R libraries not enabled'
                # for pc and fges, use their output directly
                algo = inputs["algorithm"][i]
                if algo in ["pc", "fges"]:
                    cpdag_est = W_est!=0
                    r_B_est = ro.r.matrix(cpdag_est, nrow=nr, ncol=nc)
                    # MEC-SID
                    ro.r("""mec_sid <- function(b_true, b_est){
                        res = structIntervDist(b_true, b_est)
                        return(c(res$sid, res$sidLowerBound, res$sidUpperBound))}""")
                    r_mec_sid_fun = ro.globalenv["mec_sid"]
                    mec_sid_res = list(r_mec_sid_fun(r_B_true, r_B_est))
                    # MEC-SHD
                    ro.r("""dag2cpdag <- function(x){return(pcalg::dag2cpdag(x))}""")
                    r_dag2cpdag = ro.globalenv["dag2cpdag"]
                    cpdag_true = np.array(r_dag2cpdag(r_B_true))
                    mec_shd = utils.shd_cpdag(cpdag_true, cpdag_est)
                else:
                    r_B_est = ro.r.matrix(B_est, nrow=nr, ncol=nc)
                    # MEC-SID
                    ro.r("""mec_sid <- function(b_true, b_est){
                        res = structIntervDist(b_true, pcalg::dag2cpdag(b_est))
                        return(c(res$sid, res$sidLowerBound, res$sidUpperBound))}""")
                    r_mec_sid_fun = ro.globalenv["mec_sid"]
                    mec_sid_res = list(r_mec_sid_fun(r_B_true, r_B_est))
                    # MEC-SHD
                    ro.r("""dag2cpdag <- function(x){return(pcalg::dag2cpdag(x))}""")
                    r_dag2cpdag = ro.globalenv["dag2cpdag"]
                    cpdag_true = np.array(r_dag2cpdag(r_B_true))
                    cpdag_est = np.array(r_dag2cpdag(r_B_est))
                    mec_shd = utils.shd_cpdag(cpdag_true, cpdag_est)

                results["mec_sid"].append(mec_sid_res[0])
                results["mec_sid_lower"].append(mec_sid_res[1])
                results["mec_sid_upper"].append(mec_sid_res[2])
                results["mec_shd"].append(mec_shd)

            else:
                results["mec_sid"].append(-1)
                results["mec_sid_lower"].append(-1)
                results["mec_sid_upper"].append(-1)
                results["mec_shd"].append(-1)

            results["was_dag"].append(was_dag)
            results["effective_thresholding"].append(effective_thresholding)
            results["w_threshold"].append(w_threshold)
        
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values(by=["algorithm"])
        print(f"Writing {thresholding}.csv")
        res_df.to_csv(os.path.join(self.output_folder, f"{thresholding}_{str(self.threshold)}.csv"), index=False)


if __name__ == "__main__":
    """ Evaluation settings for cluster experiments """
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Visualize results")

    parser.add_argument("--exp_base_name", type=str, help="experiment main folder", required=True)
    parser.add_argument("--exp_name", type=str, help="type (e.g. raw/normalized)", required=True)
    parser.add_argument("--base_dir", type=str, help="data input dir", default="src/experiments")
    parser.add_argument("--overwrite_prev", action="store_true", default=False, help="Overwrite _eval directory")
    parser.add_argument("--threshold", type=str, help="threshold for thresholding algos", required=True)
    parser.add_argument("--type", type=str, help="standard/favourable", required=True)
    parser.add_argument("--MEC", action="store_true", default=False)
    
    args = parser.parse_args()    
    ev = Evaluator(args.exp_name, os.path.join(args.base_dir, args.exp_base_name), args.overwrite_prev, float(args.threshold), MEC=args.MEC)
    ev.evaluate(thresholding=args.type, eligible=utils.special_thres())