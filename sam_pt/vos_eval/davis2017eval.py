"""
This script is a modified version of the original DAVIS 2017 evaluation script from:
https://github.com/davisvideochallenge/davis2017-evaluation/blob/ac7c43fca936f9722837b7fbd337d284ba37004b/evaluation_method.py

Usage:
```
python -m sam_pt.vos_eval.davis2017eval \
  --results_path /srv/beegfs02/scratch/visobt4s/data/3d_point_tracking/sampt_outputs/SegGPT--D17-val--in-sampt-env_D17_val_72_2023.11.09_15.52.53/eval_D17_val \
  --davis_path data/DAVIS/2017/trainval \
  --set val \
  --task semi-supervised \
  --year 2017
```
"""

import argparse
import os
import sys
from time import time
from typing import Union

import numpy as np
import pandas as pd
from davis2017.evaluation import DAVISEvaluation


class Davis2017Evaluator:
    def __init__(self, results_path: str, davis_path: str, set: str = "val", task: str = "semi-unsupervised",
                 year: str = '2017', sequences: Union[str, list] = "all", ):
        """
        :param results_path: Path to the folder containing the sequences folders.
        :param davis_path: Path to the DAVIS folder containing the `JPEGImages`, `Annotations`, `ImageSets`,
                           `Annotations_unsupervised` folders.
        :param set: Subset to evaluate the results.
        :param task: Task to evaluate the results.
        :param year: DAVIS dataset year.
        :param sequences: List of sequences to evaluate. If "all", evaluate all sequences.
        """
        assert set in ['val', 'test-dev', 'test-challenge']
        assert task in ['semi-supervised', 'unsupervised']

        self.davis_path = davis_path
        self.set = set
        self.task = task
        self.year = year
        self.sequences = sequences
        self.results_path = results_path

    def evaluate(self):
        time_start = time()
        csv_name_global = f'global_results-{self.set}.csv'
        csv_name_per_sequence = f'per-sequence_results-{self.set}.csv'

        # Check if the method has been evaluated before, if so read the results, otherwise compute the results
        csv_name_global_path = os.path.join(self.results_path, csv_name_global)
        csv_name_per_sequence_path = os.path.join(self.results_path, csv_name_per_sequence)
        if os.path.exists(csv_name_global_path) and os.path.exists(csv_name_per_sequence_path):
            print('Using precomputed results...')
            table_g = pd.read_csv(csv_name_global_path)
            table_seq = pd.read_csv(csv_name_per_sequence_path)
        else:
            print(f'Evaluating sequences for the {self.task} task...')
            # Create dataset and evaluate
            dataset_eval = DAVISEvaluation(davis_root=self.davis_path, task=self.task, gt_set=self.set, year=self.year,
                                           sequences=self.sequences)
            metrics_res = dataset_eval.evaluate(self.results_path)
            J, F = metrics_res['J'], metrics_res['F']

            # Generate dataframe for the general results
            g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
            final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
            g_res = np.array(
                [final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                 np.mean(F["D"])])
            g_res = np.reshape(g_res, [1, len(g_res)])
            table_g = pd.DataFrame(data=g_res, columns=g_measures)
            with open(csv_name_global_path, 'w') as f:
                table_g.to_csv(f, index=False, float_format="%.3f")
            print(f'Global results saved in {csv_name_global_path}')

            # Generate a dataframe for the per sequence results
            seq_names = list(J['M_per_object'].keys())
            seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
            J_per_object = [J['M_per_object'][x] for x in seq_names]
            F_per_object = [F['M_per_object'][x] for x in seq_names]
            table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
            with open(csv_name_per_sequence_path, 'w') as f:
                table_seq.to_csv(f, index=False, float_format="%.3f")
            print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

        # Print the results
        sys.stdout.write(f"--------------------------- Global results for {self.set} ---------------------------\n")
        print(table_g.to_string(index=False))
        sys.stdout.write(f"\n---------- Per sequence results for {self.set} ----------\n")
        print(table_seq.to_string(index=False))
        total_time = time() - time_start
        sys.stdout.write('\nTotal time:' + str(total_time))

        return table_g, table_seq


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a method on the DAVIS 2017 dataset')
    parser.add_argument('--results_path', type=str, required=True,
                        help='Path to the folder containing the sequences folders.')
    parser.add_argument('--davis_path', type=str, required=True,
                        help='Path to the DAVIS folder containing the `JPEGImages`, `Annotations`, `ImageSets`, '
                             '`Annotations_unsupervised` folders.')
    parser.add_argument('--set', type=str, default='val', choices=['val', 'test-dev', 'test-challenge'],
                        help='Subset to evaluate the results.')
    parser.add_argument('--eval_only_on_the_sequences_present_in_the_results', action='store_true',
                        help='If True, evaluate only on the sequences present in the results folder.')
    parser.add_argument('--task', type=str, default='semi-supervised', choices=['semi-supervised', 'unsupervised'],
                        help='Task to evaluate the results.')
    parser.add_argument("--year", type=str, help="Davis dataset year (default: 2017)", default='2017',
                        choices=['2016', '2017', '2019'])

    args = parser.parse_args()

    sequences = 'all'
    if args.eval_only_on_the_sequences_present_in_the_results:
        assert os.path.exists(args.results_path)
        sequences = sorted(os.listdir(args.results_path))
        sequences = [s for s in sequences if s != "overlapping" and "." not in s]
        print(f"Evaluating only on the sequences present in the results folder: {sequences}")

    evaluator = Davis2017Evaluator(args.results_path, args.davis_path, args.set, args.task, args.year, sequences)
    evaluator.evaluate()
