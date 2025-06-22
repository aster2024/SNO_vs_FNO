import numpy as np
import torch
import argparse
import os
from scipy.fft import fft, ifft, fftfreq
import h5py
from tqdm import tqdm


class PDESolver1D:
    """
    Base class for 1D PDE solvers using spectral methods.
    The class handles the spatial grid, wavenumbers, and spectral differentiation.
    """

    def __init__(self, nx, L, dt):
        """
        Initialize solver parameters.

        Args:
            nx (int): Number of spatial grid points.
            L (float): Domain length [0, L].
            dt (float): Time step.
        """
        self.nx = nx
        self.L = L
        self.dt = dt

        # Spatial grid
        self.x = np.linspace(0, L, nx, endpoint=False)
        self.dx = L / nx

        # Wavenumbers for spectral differentiation
        self.k = 2 * np.pi * fftfreq(nx, d=self.dx)
        self.k2 = self.k ** 2
        self.k3 = self.k ** 3
        self.k4 = self.k ** 4

    def spectral_derivative(self, u, order=1):
        """
        Compute spectral derivative of a given order using FFT.

        Args:
            u (np.ndarray): The function to differentiate.
            order (int): The order of the derivative.

        Returns:
            np.ndarray: The computed derivative.
        """
        u_hat = fft(u)
        if order == 1:
            return np.real(ifft(1j * self.k * u_hat))
        elif order == 2:
            return np.real(ifft(-self.k2 * u_hat))
        elif order == 3:
            return np.real(ifft(-1j * self.k3 * u_hat))
        elif order == 4:
            return np.real(ifft(self.k4 * u_hat))
        else:
            raise ValueError(f"Derivative order {order} not supported")


class IMEXRK2Solver(PDESolver1D):
    """
    A generic PDE solver using a second-order Implicit-Explicit Runge-Kutta (IMEX-RK2) scheme.
    This method is stable and relatively simple to implement.
    The PDE is assumed to be in the form: u_t = L(u) + N(u), where L is a linear operator
    and N is a nonlinear operator.
    """

    def _precompute_imex_coeffs(self):
        """
        Precomputes the integrating factor for the implicit part of the scheme.
        The implicit part is handled with a Crank-Nicolson-like approach.
        """
        # The factor comes from rearranging u_n+1 = u_n + dt/2 * (L(u_n) + L(u_{n+1})) + ...
        # which in Fourier space is û_n+1 = û_n + dt/2 * (L_op*û_n + L_op*û_{n+1}) + ...
        # (1 - dt/2 * L_op)û_{n+1} = (1 + dt/2 * L_op)û_n + ...
        self.imex_factor_numerator = 1 + self.dt / 2 * self.L_op
        self.imex_factor_denominator = 1 - self.dt / 2 * self.L_op

    def solve(self, u0, t_end):
        """
        Solves the PDE using the IMEX-RK2 scheme.

        Args:
            u0 (np.ndarray): Initial condition.
            t_end (float): Final time.

        Returns:
            np.ndarray: Solution at t=t_end.
        """
        u = u0.copy()
        u_hat = fft(u)
        t = 0

        while t < t_end:
            # Stage 1: Predictor step (explicit Euler for nonlinear part)
            N0 = self.rhs(np.real(ifft(u_hat)))
            u_star = u + self.dt * N0
            N_star = self.rhs(u_star)

            # Stage 2: Corrector step
            # Combine the implicit and explicit parts
            N_hat_avg = (fft(N0) + fft(N_star)) / 2

            # Update in Fourier space
            u_hat_new_numerator = self.imex_factor_numerator * u_hat + self.dt * N_hat_avg
            u_hat = u_hat_new_numerator / self.imex_factor_denominator

            # Update real-space solution for next loop's RHS calculation
            u = np.real(ifft(u_hat))

            t += self.dt

        return u


class BurgersSolver(IMEXRK2Solver):
    """
    Solver for 1D Burgers equation: ∂u/∂t = -u∂u/∂x + ν∂²u/∂x²
    """

    def __init__(self, nx, L, dt, nu=0.01):
        super().__init__(nx, L, dt)
        self.nu = nu
        # Linear operator in Fourier space: L = -nu * k^2
        self.L_op = -self.nu * self.k2
        self._precompute_imex_coeffs()

    def rhs(self, u):
        """Computes the nonlinear part of the Burgers equation: N(u) = -u * u_x."""
        return -u * self.spectral_derivative(u, order=1)


class KdVSolver(IMEXRK2Solver):
    """
    Solver for 1D KdV equation: ∂u/∂t = -6u∂u/∂x - ∂³u/∂x³
    """

    def __init__(self, nx, L, dt):
        super().__init__(nx, L, dt)
        # Linear operator in Fourier space: L = -(-i*k^3) = i*k^3
        self.L_op = 1j * self.k3
        self._precompute_imex_coeffs()

    def rhs(self, u):
        """Computes the nonlinear part of the KdV equation: N(u) = -6u * u_x."""
        return -6 * u * self.spectral_derivative(u, order=1)


class KSSolver(IMEXRK2Solver):
    """
    Solver for 1D Kuramoto-Sivashinsky equation: ∂u/∂t = -u∂u/∂x - ∂²u/∂x² - ∂⁴u/∂x⁴
    """

    def __init__(self, nx, L, dt):
        super().__init__(nx, L, dt)
        # CORRECTED linear operator in Fourier space: L = k^2 - k^4
        self.L_op = self.k2 - self.k4
        self._precompute_imex_coeffs()

    def rhs(self, u):
        """Computes the nonlinear part of the KS equation: N(u) = -u * u_x."""
        return -u * self.spectral_derivative(u, order=1)


def generate_initial_conditions(nx, L, n_samples, equation_type, seed=42):
    """
    Generates random initial conditions for different PDEs.
    (This function remains unchanged)
    """
    np.random.seed(seed)
    x = np.linspace(0, L, nx, endpoint=False)
    initial_conditions = []

    for _ in range(n_samples):
        if equation_type == 'burgers':
            if np.random.rand() < 0.5:
                center = np.random.uniform(0, L)
                width = np.random.uniform(0.1, 0.5) * L
                amplitude = np.random.uniform(0.5, 2.0)
                d = (x - center + L / 2) % L - L / 2
                u0 = amplitude * np.exp(-d**2/(2*width**2))
            else:
                n_modes = np.random.randint(1, 4)
                u0 = np.zeros_like(x)
                for _ in range(n_modes):
                    k = np.random.randint(1, 6)
                    amp = np.random.uniform(0.2, 1.0)
                    phase = np.random.uniform(0, 2 * np.pi)
                    u0 += amp * np.sin(2 * np.pi * k * x / L + phase)

        elif equation_type == 'kdv':
            if np.random.rand() < 0.7:
                center = np.random.uniform(0, L)
                amplitude = np.random.uniform(0.5, 3.0)
                width = np.random.uniform(0.1, 0.3) * L
                u0 = amplitude / np.cosh((x - center) / width) ** 2
            else:
                n_solitons = np.random.randint(2, 4)
                u0 = np.zeros_like(x)
                for _ in range(n_solitons):
                    center = np.random.uniform(0, L)
                    amplitude = np.random.uniform(0.3, 1.5)
                    width = np.random.uniform(0.05, 0.2) * L
                    u0 += amplitude / np.cosh((x - center) / width) ** 2

        elif equation_type == 'ks':
            n_modes = np.random.randint(3, 8)
            u0 = np.zeros_like(x)
            for _ in range(n_modes):
                k = np.random.randint(1, 6)
                amp = np.random.uniform(0.1, 0.5)
                phase = np.random.uniform(0, 2 * np.pi)
                u0 += amp * np.sin(2 * np.pi * k * x / L + phase)
            u0 += 0.1 * np.random.randn(nx)

        # Check if nan or inf values are present
        if np.any(np.isnan(u0)) or np.any(np.isinf(u0)):
            print("Warning: NaN or Inf detected in initial condition. Regenerating...")
            continue

        initial_conditions.append(u0)

    return np.array(initial_conditions)


def generate_dataset(args):
    """
    Generates dataset for the specified PDE.
    (This function remains mostly unchanged)
    """
    print(f"Generating {args.equation} dataset...")
    print(f"Domain: [0, {args.L}], Grid points: {args.nx}")
    print(f"Time step: {args.dt}, Final time: {args.t_end}")
    print(f"Training samples: {args.n_train}, Test samples: {args.n_test}")
    print(f"Using a simple and stable IMEX-RK2 solver.")

    # Create solver
    if args.equation == 'burgers':
        solver = BurgersSolver(args.nx, args.L, args.dt, args.nu)
    elif args.equation == 'kdv':
        solver = KdVSolver(args.nx, args.L, args.dt)
    elif args.equation == 'ks':
        solver = KSSolver(args.nx, args.L, args.dt)
    else:
        raise ValueError(f"Unknown equation: {args.equation}")

    # Generate initial conditions
    train_ic = generate_initial_conditions(args.nx, args.L, args.n_train, args.equation, seed=args.seed)
    test_ic = generate_initial_conditions(args.nx, args.L, args.n_test, args.equation, seed=args.seed + 1000)

    # Solve PDEs
    print("Solving training samples...")
    train_solutions = []
    for u0 in tqdm(train_ic, desc="Training samples"):
        u_final = solver.solve(u0, args.t_end)
        train_solutions.append(u_final)

        # Check for NaN or Inf in the solution
        if np.any(np.isnan(u_final)) or np.any(np.isinf(u_final)):
            print("Warning: NaN or Inf detected in solution. Skipping this sample.")
            continue

    train_solutions = np.array(train_solutions)

    print("Solving test samples...")
    test_solutions = []
    for u0 in tqdm(test_ic, desc="Test samples"):
        u_final = solver.solve(u0, args.t_end)
        test_solutions.append(u_final)

        # Check for NaN or Inf in the solution
        if np.any(np.isnan(u_final)) or np.any(np.isinf(u_final)):
            print("Warning: NaN or Inf detected in solution. Skipping this sample.")
            continue

    test_solutions = np.array(test_solutions)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save data
    filename = f"{args.equation}_nx{args.nx}_t{args.t_end}_IMEXRK2.h5"
    filepath = os.path.join(args.output_dir, filename)

    print(f"Saving data to {filepath}")
    with h5py.File(filepath, 'w') as f:
        f.attrs['equation'] = args.equation
        f.attrs['solver'] = 'IMEX-RK2'
        f.attrs['nx'] = args.nx
        f.attrs['L'] = args.L
        f.attrs['dt'] = args.dt
        f.attrs['t_end'] = args.t_end
        f.attrs['n_train'] = args.n_train
        f.attrs['n_test'] = args.n_test
        if args.equation == 'burgers':
            f.attrs['nu'] = args.nu

        f.create_dataset('x', data=solver.x)
        f.create_dataset('train_ic', data=train_ic)
        f.create_dataset('train_solutions', data=train_solutions)
        f.create_dataset('test_ic', data=test_ic)
        f.create_dataset('test_solutions', data=test_solutions)

    print(f"Dataset generation completed! Saved to: {filepath}")
    print(f"Training set shape: {train_solutions.shape}")
    print(f"Test set shape: {test_solutions.shape}")


def main():
    parser = argparse.ArgumentParser(description='Generate 1D PDE datasets using a robust IMEX-RK2 scheme')

    parser.add_argument('--equation', type=str, choices=['burgers', 'kdv', 'ks'],
                        default='burgers', help='PDE to solve')
    parser.add_argument('--nx', type=int, default=256,
                        help='Number of spatial grid points')
    parser.add_argument('--L', type=float, default=2 * np.pi,
                        help='Domain length [0, L]')
    parser.add_argument('--dt', type=float, default=0.00001,
                        help='Time step')
    parser.add_argument('--t_end', type=float, default=0.1,
                        help='Final time')
    parser.add_argument('--n_train', type=int, default=2000,
                        help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=1000,
                        help='Number of test samples')
    parser.add_argument('--nu', type=float, default=0.01,
                        help='Viscosity for Burgers equation')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()
    generate_dataset(args)


if __name__ == '__main__':
    main()
