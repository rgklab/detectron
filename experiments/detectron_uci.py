from __future__ import annotations

from random import sample

import numpy as np
import pandas as pd
import xgboost as xgb

from data.sample_data.uci import uci_heart_xgb, uci_heart_numpy
from models.pretrained import xgb_trained_on_uci_heart
from utils.detectron.modules import EarlyStopper
import torch

DEFAULT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'nthread': 4,
    'tree_method': 'gpu_hist'
}

BASE_MODEL = xgb_trained_on_uci_heart(seed=0)


def detectron_tst(train: tuple[np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray],
                  q: tuple[np.ndarray, np.ndarray], ensemble_size=10,
                  xgb_params=DEFAULT_PARAMS, base_model=BASE_MODEL, num_rounds=10,
                  patience=3):
    record = []

    # gather the data
    train_data, train_labels = train
    val_data, val_labels = val
    q_data, q_labels = q

    # store the test data
    N = len(q_data)
    q_labeled = xgb.DMatrix(q_data, label=q_labels)

    # evaluate the base model on the test data
    q_pseudo_probabilities = base_model.predict(q_labeled)
    q_pseudo_labels = q_pseudo_probabilities > 0.5

    # create the weighted dataset for training the detectron
    pq_data = xgb.DMatrix(
        data=np.concatenate([train_data, q_data]),
        label=np.concatenate([train_labels, 1 - q_pseudo_labels]),
        weight=np.concatenate(
            [np.ones_like(train_labels), 1 / (N + 1) * np.ones(N)]
        )
    )

    # set up the validation data
    val_dmatrix = xgb.DMatrix(val_data, val_labels)
    evallist = [(val_dmatrix, 'eval')]

    # evaluate the base model on test and auc data
    record.append({
        'ensemble_idx': 0,
        'val_auc': float(base_model.eval(val_dmatrix).split(':')[1]),
        'test_auc': float(base_model.eval(q_labeled).split(':')[1]),
        'rejection_rate': 0,
        # The parameter below is needed to calculate the AUC and TPR of the detectron entropy test!
        'logits': base_model.predict(q_labeled,output_margin = True), 
        'test_probabilities': q_pseudo_probabilities,
        'count': N
    })
    stopper = EarlyStopper(patience=patience, mode='min')
    stopper.update(N)

    # train the ensemble
    for i in range(1, ensemble_size + 1):
        # train the next model in the ensemble
        xgb_params.update({'seed': i})
        detector = xgb.train(xgb_params, pq_data, num_rounds, evals=evallist, verbose_eval=False)

        # evaluate the detector on the test data
        q_unlabeled = xgb.DMatrix(q_data)
        mask = ((detector.predict(q_unlabeled) > 0.5) == q_pseudo_labels)

        # filter data to exclude the not rejected samples
        q_data = q_data[mask]
        q_pseudo_labels = q_pseudo_labels[mask]
        n = len(q_data)

        # log the results for this model
        record.append({'ensemble_idx': i,
                       'val_auc': float(detector.eval(val_dmatrix).split(':')[1]),
                       'test_auc': float(detector.eval(q_labeled).split(':')[1]),
                       'rejection_rate': 1 - n / N,
                        # The parameter below is needed to calculate the AUC and TPR of the detectron entropy test!
                       'logits':  detector.predict(q_labeled,output_margin = True),    
                       'test_probabilities': detector.predict(q_labeled),
                       'count': n})

        # break if no more data
        if n == 0:
            print(f'Converged to a rejection rate of 1 after {i} models')
            break

        if stopper.update(n):
            print(f'Early stopping: Converged after {i} models')
            break

        # update the training matrix
        pq_data = xgb.DMatrix(
            data=np.concatenate([train_data, q_data]),
            label=np.concatenate([train_labels, 1 - q_pseudo_labels]),
            weight=np.concatenate(
                [np.ones_like(train_labels), 1 / (n + 1) * np.ones(n)]
            )
        )

    return record


if __name__ == '__main__':
    DATA_NUMPY = uci_heart_numpy()
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--seeds', type=int, default=100)
    parser.add_argument('--samples', default=[10, 20, 50], nargs='+')
    parser.add_argument('--splits', default=['p', 'q'], nargs='+')
    parser.add_argument('--resume', default=False, action='store_true')
    args = parser.parse_args()

    if os.path.exists(run_dir := os.path.join('results', args.run_name)) and not args.resume:
        raise ValueError(f'Run name <{args.run_name}> already exists')
    elif os.path.exists(run_dir) and args.resume:
        print(f'Resuming run <{args.run_name}>')
    else:
        os.makedirs(run_dir)
        print(f'Directory created for run: {run_dir}')

    n_runs = len(args.samples) * len(args.splits) * args.seeds
    count = 0

    print(f'Staring {n_runs} runs')

    train = DATA_NUMPY['train_data'], DATA_NUMPY['train_labels'][0]
    print('Train Eval: ', BASE_MODEL.eval(xgb.DMatrix(*train)))
    val = DATA_NUMPY['val_data'], DATA_NUMPY['val_labels'][0]
    print('Val Eval: ', BASE_MODEL.eval(xgb.DMatrix(*val)))

    for N in map(int, args.samples):
        for dataset_name in args.splits:
            for seed in range(args.seeds):
                # collect either p or q data and filter it to size N using random seed
                if dataset_name == 'p':
                    q = DATA_NUMPY['iid_test_data'], DATA_NUMPY['iid_test_labels'][0]
                else:
                    q = DATA_NUMPY['ood_test_data'], DATA_NUMPY['ood_test_labels'][0]

                # randomly sample N elements from q
                idx = np.random.RandomState(seed).permutation(len(q[0]))[:N]
                q = q[0][idx, :], q[1][idx]
                res = detectron_tst(train=train, val=val, q=q)

                for r in res:
                    r.update({'seed': seed, 'N': N, 'dataset': dataset_name})
                    for k, v in r.items():
                        if isinstance(v, np.ndarray):
                            r[k] = torch.from_numpy(v)
                print(f'{dataset_name} Rejection rate {N=}: {res[-1]["rejection_rate"]:.2f}')
                torch.save(res, f'results/{args.run_name}/test_{seed}_{dataset_name}_{N}.pt')

                count += 1
                print(f'Run {count}/{n_runs} complete')
