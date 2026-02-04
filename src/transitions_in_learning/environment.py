from dataclasses import field
from enum import Enum

import jax
from flax import struct
from gymnax.environments.environment import (
    Environment,
    EnvState as BaseEnvState,
    EnvParams as BaseEnvParams,
)
from gymnax.environments import spaces
import jax.numpy as jnp

PAIN_DELAY = 1
PAIN_DURATION = 2
PAIN_BUFFER_SIZE = PAIN_DELAY + PAIN_DURATION + 2


class ResourceType(Enum):
    EMPTY = 0
    CARDBOARD = 1
    FOOD = 2
    POISON = 3


class Action(Enum):
    NO_OP = 0
    OPEN_MOUTH = 1
    CLOSE_MOUTH = 2


@struct.dataclass
class EnvState(BaseEnvState):
    current_resource: jax.Array
    energy: jax.Array
    mouth_open: jax.Array = field(default_factory=lambda: jnp.array(False))
    pain_level: jax.Array = field(default_factory=lambda: jnp.array(0.0))
    pain_delta_buffer: jax.Array = field(
        default_factory=lambda: jnp.zeros((PAIN_BUFFER_SIZE))
    )
    pain_ptr: jax.Array = field(default_factory=lambda: jnp.array(0, dtype=jnp.int32))


@struct.dataclass
class EnvParams(BaseEnvParams):
    initial_energy: float = 5.0
    food_value: float = 1.0
    poison_value: float = -1.0
    action_cost: float = -0.01
    initial_resource_probs: jax.Array = field(
        default_factory=lambda: jnp.array([0.25, 0.25, 0.25, 0.25])
    )
    clumpiness: float = 0.5  # between 0 and 1


class CustomEnv(Environment[EnvState, EnvParams]):
    """Custom Environment Example."""

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=100, food_value=1.0, poison_value=-1.0)

    def sample_resource(
        self, key: jax.Array, prev_resource: jax.Array, params: EnvParams
    ) -> jax.Array:
        """Sample a random resource type."""
        rho = jnp.where(prev_resource == -1, 0.0, params.clumpiness)
        tmp = jnp.eye(len(ResourceType))[prev_resource]
        p = (params.initial_resource_probs * (1 - rho)) + (tmp * rho)
        return jax.random.choice(key, 4, p=p)

    def get_obs(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Get observation from the environment state."""
        resource_one_hot = jax.nn.one_hot(state.current_resource, len(ResourceType))
        pain_signal = jnp.where(state.pain_level > 0.0, 1.0, 0.0)
        return jnp.concatenate([jnp.array([pain_signal]), resource_one_hot])

    def observation_space(self, params: EnvParams):
        min_obs = jnp.zeros(1 + len(ResourceType))
        max_obs = jnp.ones_like(min_obs)
        return spaces.Box(low=min_obs, high=max_obs, shape=min_obs.shape)

    def action_space(self, params: EnvParams):
        return spaces.Discrete(len(Action))

    def get_energy_value(self, resource: jax.Array, params: EnvParams) -> jax.Array:
        return jnp.where(
            resource == ResourceType.FOOD.value,
            params.food_value,
            jnp.where(
                resource == ResourceType.POISON.value,
                params.poison_value,
                0.0,
            ),
        )

    def do_pain_stuff(
        self, state: EnvState, energy_change: jax.Array, params: EnvParams
    ) -> EnvState:
        # if agent just ate poison, schedule new pain
        def update_buffer(buffer):
            start = (state.pain_ptr + PAIN_DELAY) % PAIN_BUFFER_SIZE
            end = (start + PAIN_DURATION) % PAIN_BUFFER_SIZE
            buffer = buffer.at[start].add(1.0)
            buffer = buffer.at[end].add(-1.0)
            return buffer

        new_buffer = jax.lax.cond(
            energy_change <= params.poison_value,
            update_buffer,
            lambda d: d,
            state.pain_delta_buffer,
        )

        # apply delta scheduled for 'now'
        new_pain_level = state.pain_level + new_buffer[state.pain_ptr]
        new_buffer = new_buffer.at[state.pain_ptr].set(0.0)

        # advance pointer
        new_ptr = (state.pain_ptr + 1) % PAIN_BUFFER_SIZE

        return state.replace(
            pain_level=new_pain_level, pain_delta_buffer=new_buffer, pain_ptr=new_ptr
        )

    def reset_env(self, key, params: EnvParams) -> tuple[jax.Array, EnvState]:
        """Reset the environment to an initial state."""
        resource = self.sample_resource(key, jnp.array(-1), params)
        initial_state = EnvState(
            time=0,
            energy=jnp.array(params.initial_energy),
            current_resource=resource,
        )
        initial_obs = self.get_obs(initial_state, params)

        return initial_obs, initial_state

    def step_env(
        self, key, state: EnvState, action: int, params: EnvParams
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict]:
        """Perform a step in the environment."""
        # determine energy change
        energy_value = self.get_energy_value(state.current_resource, params)
        energy_change = jnp.where(state.mouth_open, energy_value, 0.0)
        energy_change += jnp.where(
            action != Action.NO_OP.value, params.action_cost, 0.0
        )
        new_energy = state.energy + energy_change

        # sample new resource for next timestep
        new_resource = self.sample_resource(key, state.current_resource, params)

        # determine mouth-open state for next timestep
        new_mouth_open = jnp.where(
            action == Action.OPEN_MOUTH.value,
            True,
            jnp.where(action == Action.CLOSE_MOUTH.value, False, state.mouth_open),
        )

        # get new state and observation
        new_state = EnvState(
            time=state.time + 1,
            energy=new_energy,
            current_resource=new_resource,
            mouth_open=new_mouth_open,
            pain_level=state.pain_level,
            pain_delta_buffer=state.pain_delta_buffer,
            pain_ptr=state.pain_ptr,
        )
        new_state = self.do_pain_stuff(new_state, energy_change, params)
        new_obs = self.get_obs(new_state, params)

        # determine if episode is done
        done = jnp.logical_or(
            new_state.time >= params.max_steps_in_episode, new_energy <= 0.0
        )

        return new_obs, new_state, energy_change, done, {}


def make(**kwargs) -> tuple[CustomEnv, EnvParams]:
    """Factory function to create the custom environment and its parameters."""
    env = CustomEnv()
    params = env.default_params.replace(**kwargs)
    return env, params
