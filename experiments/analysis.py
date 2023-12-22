import argparse
from glob import glob

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from scipy.stats import ks_2samp
from scipy.stats import sem

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str)
parser.add_argument('--significance', '--alpha', type=float, default=0.05)
parser.add_argument('--ensemble_size', '-n', type=int, default=5)
args = parser.parse_args()
run_name = 'results/' + args.run_name

test_p = [torch.load(x) for x in (glob(f'{run_name}/test_*_p_*.pt'))]

test_q = [torch.load(x) for x in (glob(f'{run_name}/test_*_q_*.pt'))]

for test in [test_p, test_q]:
    for d in test:
        for dd in d:
            dd['logits'] = dd['logits'].numpy()
            if 'rejection_mask' in dd:
               del dd['rejection_mask']

print(f'→ {len(test_p) + len(test_q)} runs loaded')

tp = pd.concat([pd.DataFrame(x) for x in test_p])
tq = pd.concat([pd.DataFrame(x) for x in test_q])

tp['rejection_rate'] = (1 - tp['count'] / tp['N'])
tq['rejection_rate'] = (1 - tq['count'] / tq['N'])


def n_or_last(lst, n):
    if len(lst) <= n:
        return lst.iloc[-1]
    return lst.iloc[n]


alpha = args.significance  # significance  level
n = args.ensemble_size  # ensemble size to infer with

print('→ Running Disagreement Test')
Ns = sorted(tp['N'].unique())
print('N =', ', '.join(map(str, Ns)))
for N in Ns:
    null = [n_or_last(tp.query(f'seed=={i} and N=={N}'), n + 1)['count'] for i in tp.query(f'N=={N}').seed.unique()]
    alt = [n_or_last(tq.query(f'seed=={i} and N=={N}'), n + 1)['count'] for i in tq.query(f'N=={N}').seed.unique()]
    alt = np.array(alt)
    null = np.array(null)
    X = np.arange(0, N + 2, 1)[:, None]
    tpr = (np.array(alt) < X)
    fpr = (np.array(null) < X).mean(1)

    thresh = np.where(fpr <= alpha)[0][-1]

    tpr_sem_at_5 = sem(tpr[thresh])
    tpr = tpr.mean(1)
    tpr_mean_at_5 = tpr[thresh]
    auc = np.trapz(tpr, fpr)
    print(f'TPR: {tpr_mean_at_5:.2f} \u00B1 {tpr_sem_at_5:.2f}'.replace('0.', '.'), f'AUC: {auc:.3f}', sep=' ',
          end=' | ' if N != Ns[-1] else '')

print('\n→ Running Entropy Test')
print('N =', ', '.join(map(str, Ns)))
for N in Ns:
    p_entropy = []
    for seed in tp.query(f'N=={N}').seed.unique():
        probs = tp.query(f'seed=={seed} and N=={N}').iloc[:n + 1].logits.map(lambda x: softmax(x, axis=-1)).mean()
        entropy = (-np.log(probs) * probs).sum(-1)
        p_entropy.append(entropy)
    p_entropy = np.stack(p_entropy)

    q_entropy = []
    for seed in tq.query(f'N=={N}').seed.unique():
        probs = tq.query(f'seed=={seed} and N=={N}').iloc[:n + 1].logits.map(lambda x: softmax(x, axis=-1)).mean()
        entropy = (-np.log(probs) * probs).sum(-1)
        q_entropy.append(entropy)

    q_entropy = np.stack(q_entropy)

    null_tests = []
    for i, s1 in enumerate(p_entropy):
        s2 = p_entropy[np.arange(len(p_entropy)) != i].flatten()
        null_tests.append(ks_2samp(s1, s2).pvalue)

    null = np.array(null_tests)

    alt_tests = []
    s2 = p_entropy[1:].flatten()
    for s1 in q_entropy:
        alt_tests.append(ks_2samp(s1, s2).pvalue)

    alt = np.array(alt_tests)

    if max(alt) < min(null):
        tpr_mean_at_5 = 1
        tpr_sem_at_5 = 0
        auc = 1
    else:
        X = np.arange(0, 1, 0.00001)[:, None]
        tpr = (np.array(alt) < X)
        fpr = (np.array(null) < X).mean(1)

        thresh = np.where(fpr <= alpha)[0][-1]

        tpr_sem_at_5 = sem(tpr[thresh])
        tpr = tpr.mean(1)
        tpr_mean_at_5 = tpr[thresh]
        auc = np.trapz(tpr, fpr)

    print(f'TPR: {tpr_mean_at_5:.2f} \u00B1 {tpr_sem_at_5:.2f}'.replace('0.', '.'), f'AUC: {auc:.3f}', sep=' ',
          end=' | ' if N != Ns[-1] else '')
print()
