import os
import pytorch_lightning as pl
import torch
from utils.detectron import DetectronLoader, DetectronModule, EarlyStopper

from tests.detectron.detectron import infer_labels
from data import sample_data
from data.core import split_dataset
from models import pretrained
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str)
parser.add_argument('--seeds', type=int, default=100)
parser.add_argument('--samples', default=[10, 20, 50], nargs='+')
parser.add_argument('--harmful', default=False, action='store_true')
parser.add_argument('--splits', default=['p', 'q'], nargs='+')
parser.add_argument('--train_samples', type=int, default=50000)
parser.add_argument('--val_samples', type=int, default=5000)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--ensemble_size', default=5, type=int)
parser.add_argument('--max_epochs_per_model', default=5, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--patience', default=2, type=int)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--lr', default=1e-3, type=float)
args = parser.parse_args()

# pretty print args
print('Arguments:')
for k, v in vars(args).items():
    print(f'\t{k}: {v}')
print('-' * 80)

if os.path.exists(run_dir := os.path.join('results', args.run_name)) and not args.resume:
    raise ValueError(f'Run name <{args.run_name}> already exists')
elif os.path.exists(run_dir) and args.resume:
    print(f'Resuming run <{args.run_name}>')
else:
    os.makedirs(run_dir)
    print(f'Directory created for run: {run_dir}')

load_model = lambda: pretrained.camelyon_model(seed=0, wilds=False)
p_train = sample_data.camelyon(split='train', num_samples=args.train_samples, quantized=True)
p_val = sample_data.camelyon(split='val', num_samples=args.val_samples, quantized=True)
p_test_all = sample_data.camelyon(split='test', num_samples='all')

q_split = 'harmful' if args.harmful else 'not_harmful'
q_all = sample_data.camelyon(split=q_split, num_samples='all')

test_sets = {'p': p_test_all, 'q': q_all}
base_model = load_model()
optimizers = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD}
assert args.optimizer in optimizers.keys(), f'Optimizer <{args.optimizer}> not supported'

# hyperparams ---------------------------------------------
max_epochs_per_model = args.max_epochs_per_model
optimizer = lambda params: optimizers[args.optimizer](params, lr=args.lr)
ensemble_size = args.ensemble_size
batch_size = args.batch_size
patience = args.patience
# ---------------------------------------------------------
gpus = [args.gpu]
num_workers = args.num_workers
# ---------------------------------------------------------

if os.path.exists(label_path := os.path.join(run_dir, 'pseudo_labels.pt')):
    pseudo_labels = torch.load(label_path)
    pseudo_labels_train = pseudo_labels['train']
    pseudo_labels_val = pseudo_labels['val']
    val_acc = pseudo_labels['acc']
else:
    (pseudo_labels_train, _), (pseudo_labels_val, val_acc) = infer_labels(
        model=base_model,
        dataset=(p_train, p_val),
        gpus=gpus,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=True,
        return_accuracy=True,
    )
    torch.save(dict(train=pseudo_labels_train,
                    val=pseudo_labels_val,
                    acc=val_acc),
               label_path)

runs_id = 0
runs_total = len(args.samples) * args.seeds * 2
for N in map(int, args.samples):
    for dataset_name in args.splits:
        for seed in range(args.seeds):

            # setup save paths
            runs_id += 1
            val_path = os.path.join(run_dir, f'val_{seed}_{dataset_name}_{N}.pt')
            test_path = os.path.join(run_dir, f'test_{seed}_{dataset_name}_{N}.pt')

            # look for cached results
            if os.path.exists(val_path) and os.path.exists(test_path):
                print(f'Found existing results for seed {seed}, dataset {dataset_name}, N {N}')
                continue
            else:
                print(f'Running seed {seed}, dataset {dataset_name}, N {N} (run: {runs_id}/{runs_total})')

            # set up the run parameters
            pl.seed_everything(seed)
            val_results = []
            test_results = []
            log = {'N': N, 'seed': seed, 'dataset': dataset_name, 'ensemble_idx': 0}
            count = N

            # set up the dataset
            q, _ = split_dataset(test_sets[dataset_name], N, seed)
            pseudo_labels_test = infer_labels(
                model=base_model,
                dataset=q,
                gpus=gpus,
                batch_size=N,
                num_workers=num_workers,
                verbose=True
            )
            pq_loader = DetectronLoader(p_train=p_train,
                                        p_val=p_val,
                                        q=q,
                                        p_train_pseudo_labels=pseudo_labels_train,
                                        p_val_pseudo_labels=pseudo_labels_val,
                                        q_pseudo_labels=pseudo_labels_test,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        )

            # evaluate the base model on q
            base = DetectronModule(base_model)
            pl.Trainer(gpus=gpus, logger=False, max_epochs=1).test(base, pq_loader.test_dataloader(), verbose=False)
            test_results.append(base.test_struct.to_dict() | {'count': count} | log)

            # init the val results table
            val_results.append({'accuracy': val_acc, 'rejection_rate': 0, 'accepted_accuracy': val_acc,
                                'detector_accuracy': val_acc} | log)

            # configure early stopping
            stopper = EarlyStopper(patience=patience, mode='min')
            stopper.update(count)

            # train the ensemble
            for i in range(1, ensemble_size + 1):
                log.update({'ensemble_idx': i})

                # set up the training module
                trainer = pl.Trainer(
                    gpus=gpus,
                    max_epochs=max_epochs_per_model,
                    logger=False,
                    num_sanity_val_steps=0,
                    limit_val_batches=0,
                    enable_model_summary=False
                )
                # set alpha to the suggested value in the paper
                # Note 1: we use lambda in the paper, but it is a reserved keyword, so we call it alpha here
                # Note 2: we use a custom batch sampler which slightly changes the way you compute lambda
                alpha = 1 / (len(pq_loader.train_dataloader()) * count + 1)
                detector = DetectronModule(model=load_model(),
                                           alpha=alpha)
                print(f'α = {1000 * alpha:.3f} × 10⁻³')

                # train the detectron model
                start_time = time.time()
                trainer.fit(detector, pq_loader)
                elapsed_time = time.time() - start_time
                print(f'Elapsed time: {elapsed_time:.2f} s')

                # evaluate the detectron model on the iid validation set
                trainer.test(detector, pq_loader.val_dataloader(), verbose=False)
                val_results.append(detector.test_struct.to_dict(minimal=True) | log)

                # evaluate the detectron model on the filtered q dataset
                trainer.test(detector, pq_loader.test_dataloader(), verbose=False)
                count = pq_loader.refine(~detector.test_struct.rejection_mask, verbose=True)
                test_results.append(detector.test_struct.to_dict() | {'count': count, 'runtime': elapsed_time} | log)

                # evaluate the detectron model on the full q dataset
                # (there is some redundancy here, but it makes the code much simpler)
                trainer.test(detector, pq_loader.q_dataloader(), verbose=False)
                test_results[-1].update({'logits': detector.test_struct.logits})

                # early stopping check
                if stopper.update(count):
                    print('Early stopping after', i, 'iterations')
                    break

                # check if we have enough samples to continue
                if count == 0:
                    print(f'Converged to rejection rate of 100% after {i} iterations')
                    break

            # save the results
            torch.save(val_results, os.path.join(run_dir, f'val_{seed}_{dataset_name}_{N}.pt'))
            torch.save(test_results, os.path.join(run_dir, f'test_{seed}_{dataset_name}_{N}.pt'))
