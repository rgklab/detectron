from __future__ import annotations
from sklearn.model_selection import train_test_split

import warnings
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from data.sample_data.mimic import load_data
from utils.detectron.modules import EarlyStopper
from joblib import load



# DEFAULT_PARAMS = {
#     'n_estimators': 100,
#     'criterion': 'gini',
#     'max_depth': 6,
#     'min_samples_split': 2,
#     'min_samples_leaf': 1,
#     'max_features': 'auto',
#     'n_jobs': -1,
   
# }

DEFAULT_PARAMS = {
    'bootstrap': True, 
    'ccp_alpha': 0.0, 
    'class_weight': 'balanced', 
    'criterion': 'gini', 
    'max_depth': 10, 
    'max_features': 'sqrt', 
    'min_impurity_decrease': 0.0, 
    'min_samples_leaf': 5, 
    'min_samples_split': 2, 
    'min_weight_fraction_leaf': 0.0, 
    'n_estimators': 100,
    'oob_score': False, 
    'random_state': 54288, 
    'verbose': 0, 
    'warm_start': False
    }

# Load the base model
with open("models/clf.pkl", 'rb') as f:
    BASE_MODEL = load(f)

def detectron_tst(train: tuple[np.ndarray, np.ndarray], val: tuple[np.ndarray, np.ndarray],
                  q: tuple[np.ndarray, np.ndarray], ensemble_size=10,
                  rf_params=DEFAULT_PARAMS, base_model=BASE_MODEL, patience=3):
    record = []

    # gather the data
    
    train_data, train_labels = train
    val_data, val_labels = val
    q_data, q_labels = q
    
    N = len(q_data)
    
    
    # store the test data
    q_labeled = (q_data, q_labels) 
    # print(f"Initial shapes - Train: {train_data.shape}, Val: {val_data.shape}, Q: {q_data.shape}")


    



    # Additional training:
    #base_model.fit(train_data, train_labels)


    # evaluate the base model on the test data eicu
    q_pseudo_probabilities = base_model.predict_proba(q_data)[:, 1]
    q_pseudo_labels = q_pseudo_probabilities > 0.5

    print(f"Shapes after base model - Q Pseudo Labels: {q_pseudo_labels.shape}, Q Pseudo Probabilities: {q_pseudo_probabilities.shape}")

    # create the dataset for training the detectron (concatenate training Data & q)
    pq_data = (np.concatenate([train_data, q_data]),
               np.concatenate([train_labels, 1 - q_pseudo_labels]))

    print(f"Shapes after concatenation - PQ Data: {pq_data[0].shape}, PQ Labels: {pq_data[1].shape}")

    

    
  
    # print(f"erreur: {q_pseudo_probabilities}")
    # print(f"erreur 3 {q_pseudo_labels}")
    # print(f"erreur2: {q_labels}")

    # Set up the validation data
    record.append({
        'ensemble_idx': 0,
        'val_auc': roc_auc_score(val_labels, base_model.predict_proba(val_data)[:, 1]),
        'test_auc': roc_auc_score(q_labels, base_model.predict_proba(q_labeled[0])[:, 1]),
        'rejection_rate': 0,
        'logits': base_model.predict_proba(q_labeled[0])[:, 1], # This needs to be rechecked
        'count': len(q_data)
    })
    
    stopper = EarlyStopper(patience=patience, mode='min')
    stopper.update(len(q_data))
    
    # train the ensemble
    for i in range(1, ensemble_size + 1):

        # train the next model in the ensemble
        rf_params.update({'random_state': i})
        
        # ignore the warnings concerning the Random Forest Classifier 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
             # The code that triggers the warning
            detector = RandomForestClassifier(**rf_params)
            detector.fit(*pq_data)
    

        # evaluate the detector on the test data
        q_pseudo_probabilities = detector.predict_proba(q_data)[:, 1]
        mask = ((q_pseudo_probabilities>0.5) == q_pseudo_labels)

        # filter data to exclude the not rejected samples
        q_data = q_data[mask]
        q_pseudo_labels = q_pseudo_labels[mask]
        n = len(q_data)
        
       
       
        # log the results for this model
        record.append({'ensemble_idx': i,
                       'val_auc': roc_auc_score(val_labels, detector.predict_proba(val_data)[:, 1]),
                       'test_auc': roc_auc_score(q_labels, detector.predict_proba(q_labeled[0])[:, 1]),
                       'rejection_rate': 1 - n / N,
                       'logits': detector.predict_proba(q_labeled[0])[:, 1], # This needs to be rechecked
                       'count': n})
        
        
        # break if no more data
        if n == 0:
            print(f'Converged to a rejection rate of 1 after {i} models')
            break

        if stopper.update(n):
            print(f'Early stopping: Converged after {i} models')
            break

        # update the training dataset
        pq_data = (np.concatenate([train_data, q_data]),
                   np.concatenate([train_labels, 1 - q_pseudo_labels]))

    return record


# roc_auc can't be calculated with one labeled samples so we need this function to contain the two [0,1]
def stratified_sampling(X, y, N, seed):
    X_positive, _, y_positive, _ = train_test_split(X[y == 1], y[y == 1], train_size=N, random_state=seed)
    X_negative, _, y_negative, _ = train_test_split(X[y == 0], y[y == 0], train_size=N, random_state=seed)
    q_data = np.concatenate([X_positive, X_negative])
    q_labels = np.concatenate([y_positive, y_negative])
    return q_data, q_labels

if __name__ == '__main__':
   
    # Read all Datasets
    py_train = load_data("data/df_train.csv")
    py_valid = load_data("data/df_valid.csv")
    
    # val temp
    py_mimic_2014 = load_data("data/MIMIC_2014.csv")
    py_mimic_2017 = load_data("data/MIMIC_2017.csv")
    py_mimic_2014_2017 = load_data("data/MIMIC_2014_2017.csv")

    # this will be our Q
    py_eicu = load_data("data/df_eicu.csv",True)
    
    # Calculate the number of zeros and ones
    y_true = py_eicu[1]
    num_zeros = np.sum(y_true == 0)
    num_ones = np.sum(y_true == 1)

    # print(f"Number of Zeros: {num_zeros}")
    # print(f"Number of Ones: {num_ones}")
    
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--seeds', type=int, default=100)
    parser.add_argument('--samples', default=[10,20,50], nargs='+')
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

    print(f'Starting {n_runs} runs')

    train = py_train
    print('Train Eval: ', roc_auc_score(train[1], BASE_MODEL.predict_proba(train[0])[:, 1]))
    val = py_valid
    print('Val Eval: ', roc_auc_score(val[1], BASE_MODEL.predict_proba(val[0])[:, 1]))

    for N in map(int, args.samples):
        for dataset_name in args.splits:
            for seed in range(args.seeds):
                # collect either p or q data and filter it to size N using random seed
                if dataset_name == 'p': # if it's p we take the in distribtuion data
                    q = py_mimic_2014_2017
                else:
                    q = py_eicu # else the shifted non labeled data
                

                # randomly sample N elements from q (avoid the samples that produces q_labels are all zeros!)
                # positive_sampled = False
                # while not positive_sampled:
                #     idx = np.random.RandomState(seed).permutation(len(q[0]))[:N]
                #     q = q[0][idx, :], q[1][idx]
                #     if np.sum(q[1]) > 0:
                #         positive_sampled = True
                    
                q_data, q_labels = stratified_sampling(q[0], q[1], N, seed)
                res = detectron_tst(train=train, val=val, q=(q_data,q_labels))
            
                for r in res:
                    r.update({'seed': seed, 'N': N, 'dataset': dataset_name})
                    if isinstance(r['logits'], np.ndarray):
                        r['logits'] = torch.from_numpy(r['logits'])
                print(f'{dataset_name} Rejection rate {N=}: {res[-1]["rejection_rate"]:.2f}')
                torch.save(res, f'results/{args.run_name}/test_{seed}_{dataset_name}_{N}.pt')

                count += 1
                print(f'Run {count}/{n_runs} complete')
