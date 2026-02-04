import functools
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


class MLP(nn.Module):
    features: int = 128
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nn.Dense(self.features)(x)  # shape (B, features)
        x = self.activation(x)
        return x


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=1,
        out_axes=1,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        hidden_size = rnn_state[0].shape[0]
        new_rnn_state, y = nn.GRUCell(features=hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Return zeros for the GRU hidden state instead of instantiating
        # an nn.Module (which requires a Flax scope). A GRUCell carry is
        # a tensor with shape (batch_size, hidden_size).
        return jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)


class RNNPolicy(nn.Module):
    rnn_hidden_size: int = 64
    mlp_hidden_sizes: tuple[int, ...] = (64, 64)
    num_actions: int = 3  # e.g., for categorical actions

    def init_hidden_state(self, batch_size):
        return ScannedRNN.initialize_carry(batch_size, self.rnn_hidden_size)

    @nn.compact
    def __call__(self, key, x, rnn_state, reset_rnn):
        # x shape: (B, T, obs_dim)
        batch_size, seq_len, _ = x.shape

        # Process observations with MLP
        x = x.reshape(batch_size * seq_len, -1)  # (B*T, obs_dim)
        x = MLP(features=self.rnn_hidden_size)(x)  # (B*T, rnn_hidden_size)
        x = x.reshape(batch_size, seq_len, -1)  # (B, T, rnn_hidden_size)

        # Prepare resets for RNN
        resets = reset_rnn.astype(jnp.float32)  # (B, T)

        # Apply RNN
        rnn_module = ScannedRNN()
        new_rnn_state, rnn_outputs = rnn_module(
            rnn_state, (x, resets)
        )  # (B, T, rnn_hidden_size)

        # Process RNN outputs with final MLP to get actions
        rnn_outputs = rnn_outputs.reshape(
            batch_size * seq_len, -1
        )  # (B*T, rnn_hidden_size)
        for hidden_size in self.mlp_hidden_sizes:
            rnn_outputs = nn.Dense(hidden_size)(rnn_outputs)
            rnn_outputs = nn.relu(rnn_outputs)

        logits = nn.Dense(self.num_actions)(rnn_outputs)  # (B*T, num_actions)
        actions = jax.random.categorical(key, logits)  # (B*T,)
        actions = actions.reshape(batch_size, seq_len)  # (B, T)

        return actions, new_rnn_state
