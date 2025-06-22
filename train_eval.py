import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import h5py
import argparse
import os
from tqdm import tqdm
from torch_dct import dct, idct

from models import FNO1d, SNOChebyshev1d, SNOFourier1d


class L2RelLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        true_norm = torch.norm(true, p=2, dim=1, keepdim=True)
        error_norm = torch.norm(pred - true, p=2, dim=1, keepdim=True)
        loss = error_norm / (true_norm + 1e-8)
        return loss.mean() if self.reduction == 'mean' else loss.sum()


def to_chebyshev_coeffs(f: torch.Tensor, n_coeffs: int) -> torch.Tensor:
    """
    Transforms function values to Chebyshev coefficients using torch.fft.dct.
    """
    f_coeffs = dct(f, norm='ortho')
    return f_coeffs[..., :n_coeffs]


def from_chebyshev_coeffs(f_coeffs: torch.Tensor, n_points: int) -> torch.Tensor:
    """
    Transforms Chebyshev coefficients back to function values using torch.fft.idct.
    """
    f_coeffs_real = f_coeffs.real
    padded_coeffs = torch.zeros(*f_coeffs_real.shape[:-1], n_points, dtype=f_coeffs_real.dtype,
                                device=f_coeffs_real.device)
    padded_coeffs[..., :f_coeffs_real.shape[-1]] = f_coeffs_real
    f_phys = idct(padded_coeffs, norm='ortho')
    return f_phys


def to_fourier_coeffs(f: torch.Tensor, n_coeffs: int) -> torch.Tensor:
    f_coeffs_cplx = torch.fft.rfft(f, dim=-1, norm='ortho')
    return f_coeffs_cplx[..., :n_coeffs]


def from_fourier_coeffs(f_coeffs_cplx: torch.Tensor, n_points: int) -> torch.Tensor:
    f_phys = torch.fft.irfft(f_coeffs_cplx, n=n_points, dim=-1, norm='ortho')
    return f_phys


def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.data_path}")
    with h5py.File(args.data_path, 'r') as f:
        train_ic, train_sol = torch.from_numpy(f['train_ic'][:]).float(), torch.from_numpy(
            f['train_solutions'][:]).float()
        test_ic, test_sol = torch.from_numpy(f['test_ic'][:]).float(), torch.from_numpy(f['test_solutions'][:]).float()
        nx = f.attrs['nx']

    train_ic, train_sol = train_ic.unsqueeze(-1), train_sol.unsqueeze(-1)
    test_ic, test_sol = test_ic.unsqueeze(-1), test_sol.unsqueeze(-1)

    train_loader = DataLoader(TensorDataset(train_ic, train_sol), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_ic, test_sol), batch_size=args.batch_size, shuffle=False)

    print("\nInitializing model...")
    if args.model_type == 'fno':
        model = FNO1d(in_channels=1, out_channels=1, width=args.width, modes=args.modes, depth=args.depth).to(device)
    elif args.model_type == 'sno_chebyshev':
        model = SNOChebyshev1d(in_channels=1, out_channels=1, n_coeffs=args.n_coeffs,
                               lifting_features=args.lifting_features, integral_neurons=args.integral_neurons,
                               n2_depth=args.depth).to(device)
    elif args.model_type == 'sno_fourier':
        model = SNOFourier1d(in_channels=1, out_channels=1, n_coeffs=args.n_coeffs,
                             lifting_features=args.lifting_features, integral_neurons=args.integral_neurons,
                             n2_depth=args.depth).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model_type.upper()} | Trainable Parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = L2RelLoss()

    print("\nStarting training...")
    for epoch in tqdm(range(args.epochs), desc="Training Progress"):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            x_phys, y_phys = x.squeeze(-1), y.squeeze(-1)

            if args.model_type == 'fno':
                pred = model(x)
                loss = loss_fn(pred, y)
            else:  # SNO models
                if args.model_type == 'sno_chebyshev':
                    x_coeffs_real = to_chebyshev_coeffs(x_phys, args.n_coeffs)
                    x_coeffs_cplx = x_coeffs_real.to(torch.cfloat).unsqueeze(-1)
                    pred_coeffs_cplx = model(x_coeffs_cplx).squeeze(-1)
                    pred_phys = from_chebyshev_coeffs(pred_coeffs_cplx, nx)

                elif args.model_type == 'sno_fourier':
                    x_coeffs_cplx = to_fourier_coeffs(x_phys, args.n_coeffs).unsqueeze(-1)
                    pred_coeffs_cplx = model(x_coeffs_cplx).squeeze(-1)
                    pred_phys = from_fourier_coeffs(pred_coeffs_cplx, nx)

                loss = loss_fn(pred_phys.unsqueeze(-1), y_phys.unsqueeze(-1))

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        if (epoch + 1) % 2000 == 0:  # Print training loss every 2000 epochs
            print(f"Epoch {epoch + 1:03d}/{args.epochs} | Train Loss: {train_loss:.6f}")

    print("\nTraining finished. Evaluating on the test set...")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x_phys, y_phys = x.squeeze(-1), y.squeeze(-1)

            if args.model_type == 'fno':
                pred = model(x)
                loss = loss_fn(pred, y)
            else:  # SNO models
                if args.model_type == 'sno_chebyshev':
                    x_coeffs_real = to_chebyshev_coeffs(x_phys, args.n_coeffs)
                    x_coeffs_cplx = x_coeffs_real.to(torch.cfloat).unsqueeze(-1)
                    pred_coeffs_cplx = model(x_coeffs_cplx).squeeze(-1)
                    pred_phys = from_chebyshev_coeffs(pred_coeffs_cplx, nx)
                elif args.model_type == 'sno_fourier':
                    x_coeffs_cplx = to_fourier_coeffs(x_phys, args.n_coeffs).unsqueeze(-1)
                    pred_coeffs_cplx = model(x_coeffs_cplx).squeeze(-1)
                    pred_phys = from_fourier_coeffs(pred_coeffs_cplx, nx)

                loss = loss_fn(pred_phys.unsqueeze(-1), y_phys.unsqueeze(-1))

            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Final Test L2 Rel Error: {test_loss:.6f}")

    print("\nSaving model...")
    save_path = os.path.join(args.output_dir, f"{args.model_type}_params_{num_params}_final.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train/evaluate FNO and SNO models with fair default parameters.")
    parser.add_argument('--model_type', type=str, required=True, choices=['fno', 'sno_chebyshev', 'sno_fourier'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./saved_models')
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--depth', type=int, default=3, help='Depth for both FNO and SNO (number of spectral layers)')

    # --- FAIR HYPERPARAMETER DEFAULTS ---
    # FNO defaults (approx. 209.2k params)
    parser.add_argument('--modes', type=int, default=16, help='FNO: Number of Fourier modes')
    parser.add_argument('--width', type=int, default=64, help='FNO: Feature width (scaled up for fairness)')

    # SNO defaults (approx. 209.1k params)
    parser.add_argument('--n_coeffs', type=int, default=64, help='SNO: Number of spectral coefficients')
    parser.add_argument('--lifting_features', type=int, default=64, help='SNO: Feature width in lifting/projection')
    parser.add_argument('--integral_neurons', type=int, default=256,
                        help='SNO: Number of neurons in integral layer')

    args = parser.parse_args()
    run_experiment(args)