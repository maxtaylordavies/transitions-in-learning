from functools import partial

import jax
from evosax.algorithms import SNES as ES
import optax
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transitions_in_learning.gymnax_problem import RNNGymnaxProblem as Problem
from transitions_in_learning.environment import make
from transitions_in_learning.rnn import RNNPolicy

sns.set_style("whitegrid")

base_seed = 0

num_keys = 5
num_generations = 100
population_size = 128
mlp_hidden_sizes = (32, 32)
rnn_hidden_size = 32
adam_lr = 0.01
episode_length = 200
num_rollouts = 16

policy = RNNPolicy(
    rnn_hidden_size=rnn_hidden_size,
    mlp_hidden_sizes=mlp_hidden_sizes,
    num_actions=3,
)


def make_problem_instance(env_params):
    return Problem(
        env_make_fn=make,
        policy=policy,
        episode_length=episode_length,
        num_rollouts=num_rollouts,
        use_normalize_obs=True,
        env_params=env_params,
    )


@partial(jax.jit, static_argnames=("problem",))
def run_problem_instance(key: jax.Array, problem: Problem):
    key, subkey = jax.random.split(key)
    problem_state = problem.init(key)

    key, subkey = jax.random.split(key)
    solution = problem.sample(subkey)

    es = ES(
        population_size=population_size,
        solution=solution,
        optimizer=optax.adam(learning_rate=adam_lr),
    )

    params = es.default_params

    @jax.jit
    def step(carry, key):
        state, params, problem_state = carry
        key_ask, key_eval, key_tell = jax.random.split(key, 3)

        population, state = es.ask(key_ask, state, params)

        fitness, problem_state, _ = problem.eval(key_eval, population, problem_state)

        state, metrics = es.tell(
            key_tell, population, -fitness, state, params
        )  # Minimize fitness

        return (state, params, problem_state), metrics

    key, subkey = jax.random.split(key)
    state = es.init(subkey, solution, params)

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_generations)
    _, metrics = jax.lax.scan(
        step,
        (state, params, problem_state),
        keys,
    )

    return metrics


run_problem_instance_vmapped = jax.vmap(run_problem_instance, in_axes=(0, None))

problem_dict = {
    "Empty obs": make_problem_instance(
        {"can_sense_features": False, "can_sense_pain": False}
    ),
    "Features only": make_problem_instance(
        {"can_sense_features": True, "can_sense_pain": False}
    ),
    "Pain only": make_problem_instance(
        {"can_sense_features": False, "can_sense_pain": True}
    ),
    "Full obs": make_problem_instance(
        {"can_sense_features": True, "can_sense_pain": True}
    ),
}

keys = jax.random.split(jax.random.PRNGKey(base_seed), num_keys)

results = {"seed": [], "env variant": [], "generation": [], "max fitness": []}
for name, problem in problem_dict.items():
    print(f"Running problem instance: {name}")
    metrics = run_problem_instance_vmapped(keys, problem)
    for key_idx in range(num_keys):
        fitness = np.array(-metrics["best_fitness"][key_idx]).tolist()
        generations = np.arange(len(fitness)).tolist()
        results["seed"].extend([base_seed + key_idx] * len(fitness))
        results["env variant"].extend([name] * len(fitness))
        results["generation"].extend(generations)
        results["max fitness"].extend(fitness)
results = pd.DataFrame(results)

print(results)

fig, ax = plt.subplots(figsize=(6, 3))
sns.lineplot(data=results, x="generation", y="max fitness", hue="env variant", ax=ax)
ax.set(
    title="SNES (RNN)",
    xlabel="Generations",
    ylabel="Max fitness",
)
sns.despine(ax=ax, left=True, bottom=True)
fig.tight_layout()
plt.show()
