import torch
import torch.nn as nn
import torch.fft


class ComplexSpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
            * (1.0 / in_channels)
        )

    def forward(self, x):
        batch, _, n = x.shape
        x_fft = torch.fft.rfft(x, dim=-1, norm="ortho")
        out_fft = torch.zeros(
            batch, self.out_channels, x_fft.size(-1),
            dtype=torch.cfloat, device=x.device
        )
        out_fft[:, :, :self.modes] = torch.einsum(
            "bim, iom -> bom",
            x_fft[:, :self.in_channels, :self.modes],
            self.weight
        )
        x_out = torch.fft.irfft(out_fft, n=n, dim=-1, norm="ortho")
        return x_out

class FourierLayer1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral_conv = ComplexSpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x):
        return self.pointwise(x) + self.spectral_conv(x)

class FNO1d(nn.Module):
    def __init__(self, in_channels, out_channels, width=32, modes=16, depth=4):
        super().__init__()
        self.fc0 = nn.Linear(in_channels, width)
        self.layers = nn.ModuleList([
            FourierLayer1d(width, modes) for _ in range(depth)
        ])
        self.activation = nn.GELU()
        self.fc1 = nn.Linear(width, out_channels)

    def forward(self, x):
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = self.activation(layer(x))
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x

class ComplexGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.real_activation = nn.GELU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.real_activation(z.real) + 1j * self.real_activation(z.imag)


class ComplexSNOLayer(nn.Module):
    def __init__(self, in_features, out_features, n_coeffs, integral_neurons, activation):
        super().__init__()
        self.B = nn.Parameter(torch.randn(integral_neurons, n_coeffs, dtype=torch.cfloat) * (1.0 / n_coeffs))
        self.A = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.cfloat) * (1.0 / in_features))
        self.b = nn.Parameter(torch.randn(integral_neurons, out_features, dtype=torch.cfloat))
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.einsum('rk,bkl,lm->brm', self.B, x, self.A) + self.b.unsqueeze(0)
        return self.activation(out)


class ComplexSimpleLayer(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.cfloat) * (1.0 / in_features))
        self.b = nn.Parameter(torch.randn(1, out_features, dtype=torch.cfloat))
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.einsum('bkl,lm->bkm', x, self.A) + self.b
        return self.activation(out)


class SNO1d(nn.Module):
    """
    1D Spectral Neural Operator.
    """

    def __init__(self, in_channels, out_channels, n_coeffs,
                 lifting_features=20, integral_neurons=100, n2_depth=3):
        super().__init__()

        activation = ComplexGELU()

        # N1: Lifting network (lifts to complex feature space)
        self.lift = ComplexSimpleLayer(in_channels, lifting_features, activation=activation)

        # N2: Spectral mixing network
        self.spectral_layers = nn.ModuleList()
        current_coeffs = n_coeffs
        for _ in range(n2_depth):
            layer = ComplexSNOLayer(lifting_features, lifting_features, current_coeffs, integral_neurons,
                                    activation=activation)
            self.spectral_layers.append(layer)
            # The number of "coefficients" for the next layer is the number of integral neurons from the current one
            current_coeffs = integral_neurons

        # N3: Projection network
        # This layer projects the features back down to the desired output channel dimension.
        self.project = ComplexSimpleLayer(lifting_features, out_channels,
                                          activation=lambda x: x)  # No activation on final layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N1: Lift to feature space
        x = self.lift(x)

        # N2: Apply spectral layers
        for layer in self.spectral_layers:
            x = layer(x)

        # N3: Project to output features
        x = self.project(x)

        return x


class SNOChebyshev1d(SNO1d):
    def __init__(self, in_channels, out_channels, n_coeffs, **kwargs):
        super().__init__(in_channels, out_channels, n_coeffs, **kwargs)


class SNOFourier1d(SNO1d):
    def __init__(self, in_channels, out_channels, n_coeffs, **kwargs):
        super().__init__(in_channels, out_channels, n_coeffs, **kwargs)