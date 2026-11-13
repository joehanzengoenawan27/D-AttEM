import os
import json
import time
import argparse
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import MLPDiffusion, Model
from dataset import load_dataset, get_eval, mean_std, save_imputed_dataset
from diffusion_utils import sample_step, impute_mask

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class DPWrapper(torch.nn.DataParallel):
    def forward(self, x, sigma, *args, **kwargs):
        if isinstance(sigma, torch.Tensor):
            if sigma.dim() == 0:
                sigma = sigma.unsqueeze(0).expand(x.shape[0])
            elif sigma.numel() == 1:
                sigma = sigma.view(1).expand(x.shape[0])
        return super().forward(x, sigma, *args, **kwargs)

    @property
    def sigma_min(self):
        return self.module.sigma_min

    @property
    def sigma_max(self):
        return self.module.sigma_max

    def round_sigma(self, sigma):
        return self.module.round_sigma(sigma)


def format_hms(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Missing Value Imputation')

parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--split_idx', type=int, default=0, help='Split idx.')
parser.add_argument('--max_iter', type=int, default=5, help='Maximum iteration.')
parser.add_argument('--ratio', type=str, default=30, help='Masking ratio.')
parser.add_argument('--hid_dim', type=int, default=1024, help='Hidden dimension.')
parser.add_argument('--mask', type=str, default='MCAR', help='Masking mechanisms.')
parser.add_argument('--num_trials', type=int, default=10, help='Number of sampling times.')
parser.add_argument('--num_steps', type=int, default=50, help='Number of diffusion steps.')
parser.add_argument('--d_model', type=int, default=32, help='Transformer hidden dimension.')
parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads.')
parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers.')
parser.add_argument(
    '--variant',
    type=str,
    default='D',
    choices=['base', 'base_silu', 'C', 'D', 'C.1', 'D.1', 'E'],
    help="Ablation variant: base, base_silu, C/D (3-layer MLP), C.1/D.1 (4-layer full MLP), E (per-token head).",
)

args = parser.parse_args()

# check cuda
if args.gpu != -1 and torch.cuda.is_available():
    args.device = f'cuda:{args.gpu}'
    print(f"Menggunakan GPU: {args.device}")
else:
    args.device = 'cpu'
    print("Menggunakan CPU!")


if __name__ == '__main__':

    dataname = args.dataname
    split_idx = args.split_idx
    device = args.device
    hid_dim = args.hid_dim
    mask_type = args.mask
    ratio = args.ratio
    num_trials = args.num_trials
    num_steps = args.num_steps

    if mask_type == 'MNAR':
        mask_type = 'MNAR_logistic_T2'

    train_X, test_X, ori_train_mask, ori_test_mask, train_num, test_num, train_cat_idx, test_cat_idx, train_mask, test_mask, cat_bin_num = load_dataset(
        dataname, split_idx, mask_type, ratio
    )

    mean_X, std_X = mean_std(train_X, train_mask)
    in_dim = train_X.shape[1]

    X = (train_X - mean_X) / std_X / 2
    X = torch.tensor(X)

    X_test = (test_X - mean_X) / std_X / 2
    X_test = torch.tensor(X_test)

    mask_train = torch.tensor(train_mask)
    mask_test = torch.tensor(test_mask)

    MAEs = []
    RMSEs = []
    ACCs = []
    F1s = []

    MAEs_out = []
    RMSEs_out = []
    ACCs_out = []
    F1s_out = []

    variant_suffix = '' if args.variant == 'D' else f'_{args.variant}'
    ckpt_dir = os.path.join(ROOT_DIR, 'ckpt', dataname, f'rate{ratio}', mask_type, str(split_idx), f'{num_trials}_{num_steps}{variant_suffix}')
    result_save_path = os.path.join(ROOT_DIR, 'results', dataname, f'rate{ratio}', mask_type, str(split_idx), f'{num_trials}_{num_steps}{variant_suffix}')
    os.makedirs(result_save_path, exist_ok=True)

    mask_start_time = time.time()
    runtime_log = {
        "dataname": dataname,
        "variant": args.variant,
        "split_idx": int(split_idx),
        "mask_type": mask_type,
        "ratio": str(ratio),
        "in_dim": int(in_dim),
        "num_params": 0,
        "iterations": [],
    }

    for iteration in range(args.max_iter):
        iter_start_time = time.time()

        ## M-Step: Density Estimation
        os.makedirs(f'{ckpt_dir}/{iteration}', exist_ok=True)

        print(f'iteration: {iteration} | variant: {args.variant} | split: {split_idx}')
        print(ckpt_dir)

        if iteration == 0:
            X_miss = (1. - mask_train.float()) * X
            train_data = X_miss.numpy()
        else:
            print(f'Loading X_miss from {ckpt_dir}/iter_{iteration}.npy')
            X_miss = np.load(f'{ckpt_dir}/iter_{iteration}.npy') / 2
            train_data = X_miss

        batch_size = 4096
        # num_workers=0 menghilangkan overhead pembuatan subprocess di setiap epoch
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        num_epochs = 10000 + 1

        denoise_fn = MLPDiffusion(
            d_in=in_dim,
            dim_t=hid_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            variant=args.variant,
        ).to(device)

        num_params = sum(p.numel() for p in denoise_fn.parameters())
        runtime_log["num_params"] = int(num_params)

        if iteration == 0:
            print(denoise_fn)
            print(f'Variant: {args.variant} | Parameters: {num_params:,}')

        model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=50)

        m_step_start = time.time()
        model.train()

        best_loss = float('inf')
        patience = 0
        stopped_epoch = num_epochs - 1

        pbar = tqdm(range(num_epochs), desc='Training')
        for epoch in pbar:

            batch_loss = 0.0
            len_input = 0

            for batch in train_loader:
                inputs = batch.float().to(device)
                loss = model(inputs)

                loss = loss.mean()
                batch_loss += loss.item() * len(inputs)
                len_input += len(inputs)

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping untuk mencegah lonjakan gradien pada Transformer
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            curr_loss = batch_loss / len_input
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
                best_loss = curr_loss
                patience = 0
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), f'{ckpt_dir}/{iteration}/model.pt')
            else:
                patience += 1
                if patience == 500:
                    stopped_epoch = epoch
                    print(f'Early stopping at epoch {epoch}')
                    break

            pbar.set_postfix(loss=curr_loss)

            if epoch % 1000 == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), f'{ckpt_dir}/{iteration}/model_{epoch}.pt')

        m_step_sec = time.time() - m_step_start

        ## E-Step: Missing Value Imputation
        e_in_start = time.time()
        in_dim = X.shape[1]

        denoise_fn = MLPDiffusion(
            d_in=in_dim,
            dim_t=hid_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            variant=args.variant,
        ).to(device)

        model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
        model.load_state_dict(torch.load(f'{ckpt_dir}/{iteration}/model.pt', map_location=device))
        model.eval()

        net = model.denoise_fn_D

        # 1. In-sample imputation
        rec_Xs = []

        for trial in tqdm(range(num_trials), desc='In-sample imputation'):
            X_miss = (1. - mask_train.float()) * X
            X_miss = X_miss.to(device)
            impute_X = X_miss

            num_samples, dim = X.shape[0], X.shape[1]
            rec_X = impute_mask(net, impute_X, mask_train, num_samples, dim, num_steps, device)
            # Safety clamp mencegah outlier merusak iterasi EM berikutnya
            rec_X = torch.clamp(rec_X, min=-5.0, max=5.0)

            mask_int = mask_train.to(torch.float).to(device)
            rec_X = rec_X * mask_int + impute_X * (1 - mask_int)
            rec_Xs.append(rec_X)

        rec_X = torch.stack(rec_Xs, dim=0).mean(0)

        rec_X = rec_X.cpu().numpy() * 2
        X_true = X.cpu().numpy() * 2

        np.save(f'{ckpt_dir}/iter_{iteration+1}.npy', rec_X)

        pred_X_train = rec_X.copy()
        len_num = train_num.shape[1]

        res = pred_X_train[:, len_num:] * std_X[len_num:] + mean_X[len_num:]
        pred_X_train[:, len_num:] = res

        mae, rmse, acc, f1 = get_eval(dataname, pred_X_train, X_true, train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask)
        MAEs.append(mae)
        RMSEs.append(rmse)
        ACCs.append(acc)
        F1s.append(f1)
        e_in_sec = time.time() - e_in_start

        print('in-sample', mae, rmse, 'acc:', acc, 'f1:', f1)

        # 2. Out-of-sample imputation & Penyimpanan Dataset Terimputasi pada Iterasi Terakhir
        e_out_start = time.time()
        if iteration == args.max_iter - 1:
            rec_Xs = []

            for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
                X_miss = (1. - mask_test.float()) * X_test
                X_miss = X_miss.to(device)
                impute_X = X_miss

                num_samples, dim = X_test.shape[0], X_test.shape[1]
                rec_X = impute_mask(net, impute_X, mask_test, num_samples, dim, num_steps, device)
                rec_X = torch.clamp(rec_X, min=-5.0, max=5.0)

                mask_int = mask_test.to(torch.float).to(device)
                rec_X = rec_X * mask_int + impute_X * (1 - mask_int)
                rec_Xs.append(rec_X)

            rec_X = torch.stack(rec_Xs, dim=0).mean(0)

            rec_X = rec_X.cpu().numpy() * 2
            X_true = X_test.cpu().numpy() * 2

            pred_X_test = rec_X.copy()
            len_num = train_num.shape[1]
            res = pred_X_test[:, len_num:] * std_X[len_num:] + mean_X[len_num:]
            pred_X_test[:, len_num:] = res

            mae_out, rmse_out, acc_out, f1_out = get_eval(dataname, pred_X_test, X_true, test_cat_idx, test_num.shape[1], cat_bin_num, ori_test_mask, oos=True)

            # Simpan dataset train_imputed.csv dan test_imputed.csv untuk evaluasi Downstream Task
            save_imputed_dataset(
                dataname=dataname,
                pred_X_train=pred_X_train,
                pred_X_test=pred_X_test,
                mean_X=mean_X,
                std_X=std_X,
                num_num=len_num,
                cat_bin_num=cat_bin_num,
                ori_train_mask=ori_train_mask,
                ori_test_mask=ori_test_mask,
                save_dir=result_save_path,
            )
        else:
            mae_out = 0.0
            rmse_out = 0.0
            acc_out = 0.0
            f1_out = 0.0
        e_out_sec = time.time() - e_out_start

        iter_total_sec = time.time() - iter_start_time
        runtime_log["iterations"].append({
            "iteration": int(iteration),
            "stopped_epoch": int(stopped_epoch),
            "m_step_sec": round(m_step_sec, 2),
            "e_in_sec": round(e_in_sec, 2),
            "e_out_sec": round(e_out_sec, 2),
            "iter_total_sec": round(iter_total_sec, 2),
            "iter_total_hms": format_hms(iter_total_sec),
        })

        MAEs_out.append(mae_out)
        RMSEs_out.append(rmse_out)
        ACCs_out.append(acc_out)
        F1s_out.append(f1_out)

        write_mode = 'w' if iteration == 0 else 'a+'
        with open(f'{result_save_path}/result.txt', write_mode) as f:
            f.write(f'iteration {iteration}, MAE: in-sample: {mae}, out-of-sample: {mae_out} \n')
            f.write(f'iteration {iteration}, RMSE: in-sample: {rmse}, out-of-sample: {rmse_out} \n')
            f.write(f'iteration {iteration}, ACC: in-sample: {acc}, out-of-sample: {acc_out} \n')
            f.write(f'iteration {iteration}, F1: in-sample: {f1}, out-of-sample: {f1_out} \n')
            f.write(
                f'iteration {iteration}, TIME: total={iter_total_sec:.2f}s ({format_hms(iter_total_sec)}) '
                f'| M-Step={m_step_sec:.2f}s (epoch={stopped_epoch}) | E-In={e_in_sec:.2f}s | E-Out={e_out_sec:.2f}s\n'
            )

        print('out-of-sample', mae_out, rmse_out, 'acc:', acc_out, 'f1:', f1_out)
        print(f'Iter {iteration} runtime: {format_hms(iter_total_sec)} (M-Step: {m_step_sec:.1f}s [epoch {stopped_epoch}], E-In: {e_in_sec:.1f}s, E-Out: {e_out_sec:.1f}s)')
        print(f'saving results to {result_save_path}')

    total_mask_sec = time.time() - mask_start_time
    runtime_log["total_mask_sec"] = round(total_mask_sec, 2)
    runtime_log["total_mask_hms"] = format_hms(total_mask_sec)

    with open(f'{result_save_path}/result.txt', 'a+') as f:
        f.write(f'TOTAL_MASK_RUNTIME: {total_mask_sec:.2f}s ({format_hms(total_mask_sec)}) | Parameters: {runtime_log["num_params"]}\n')

    with open(f'{result_save_path}/runtime.json', 'w') as f:
        json.dump(runtime_log, f, indent=2)

    print(f'Total mask runtime ({args.variant}, split {split_idx}): {format_hms(total_mask_sec)}')