"""
Spatiotemporal Convolutional LSTM (ConvLSTM) for Atmospheric Fluid Dynamics.
Maintains 2D spatial memory to model non-rigid cloud condensation, evaporation,
and advection thermodynamics across synthesized temporal steps.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    2D Convolutional LSTM Cell preserving spatial feature dimensions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Tuple[int, int] = (3, 3),
        bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Combined convolution for all 4 gates (input, forget, cell, output)
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        cur_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single time step.

        Args:
            x: Input tensor (B, C_in, H, W).
            cur_state: Tuple of (H_{t-1}, C_{t-1}) each of shape (B, C_hidden, H, W).

        Returns:
            Tuple of (H_t, C_t) each of shape (B, C_hidden, H, W).
        """
        b, _, h, w = x.shape
        device = x.device

        if cur_state is None:
            h_cur = torch.zeros(b, self.hidden_dim, h, w, device=device, dtype=x.dtype)
            c_cur = torch.zeros(b, self.hidden_dim, h, w, device=device, dtype=x.dtype)
        else:
            h_cur, c_cur = cur_state

        # Concatenate input and previous hidden state along channel dimension
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)

        # Split into input, forget, cell, output gates
        i_gate, f_gate, c_candidate, o_gate = torch.split(gates, self.hidden_dim, dim=1)

        i_t = torch.sigmoid(i_gate)
        f_t = torch.sigmoid(f_gate)
        c_tilde = torch.tanh(c_candidate)
        o_t = torch.sigmoid(o_gate)

        # State updates
        c_next = f_t * c_cur + i_t * c_tilde
        h_next = o_t * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(
        self,
        batch_size: int,
        spatial_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes zero state tensors for (H, C).
        """
        h, w = spatial_size
        return (
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device, dtype=dtype),
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device, dtype=dtype),
        )


class ConvLSTM(nn.Module):
    """
    Multi-layer Convolutional LSTM architecture.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        kernel_size: Tuple[int, int] = (3, 3),
        num_layers: int = 2,
        batch_first: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.batch_first = batch_first

        cell_list = []
        for i in range(num_layers):
            cur_in_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_in_dim,
                    hidden_dim=hidden_dims[i],
                    kernel_size=kernel_size,
                    bias=bias,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Processes temporal sequence of spatial tensors.

        Args:
            x: Input sequence tensor.
               If batch_first=True: shape (B, T, C, H, W).
               If batch_first=False: shape (T, B, C, H, W).
            hidden_state: Optional list of (H, C) tuples per layer.

        Returns:
            layer_output: Output sequence from the top layer (B, T, C_out, H, W).
            last_state_list: List of final (H_T, C_T) tuples for all layers.
        """
        if not self.batch_first:
            # Transpose to (B, T, C, H, W)
            x = x.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = x.shape
        device = x.device

        if hidden_state is None:
            hidden_state = self._init_hidden(b, (h, w), device, x.dtype)

        cur_input = x
        last_state_list = []

        for layer_idx, cell in enumerate(self.cell_list):
            h_cur, c_cur = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                x_t = cur_input[:, t, :, :, :]
                h_cur, c_cur = cell(x_t, (h_cur, c_cur))
                output_inner.append(h_cur)

            # Stack along time dimension: (B, T, C_hidden, H, W)
            layer_output = torch.stack(output_inner, dim=1)
            cur_input = layer_output
            last_state_list.append((h_cur, c_cur))

        if not self.batch_first:
            layer_output = layer_output.permute(1, 0, 2, 3, 4)

        return layer_output, last_state_list

    def _init_hidden(
        self,
        batch_size: int,
        spatial_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        init_states = []
        for cell in self.cell_list:
            init_states.append(cell.init_hidden(batch_size, spatial_size, device, dtype))
        return init_states
