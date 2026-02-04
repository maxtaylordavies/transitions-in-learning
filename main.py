import jax
from evosax.problems.networks import MLP, categorical_output_fn
from evosax.algorithms import SNES as ES
import optax
import matplotlib.pyplot as plt

from transitions_in_learning.gymnax_problem import GymnaxProblem as Problem
from transitions_in_learning.environment import make

key = jax.random.PRNGKey(0)

num_generations = 64
population_size = 128
mlp_hidden_sizes = (64, 64)
adam_lr = 0.01

policy = MLP(
    layer_sizes=(*mlp_hidden_sizes, 3),  # 3 actions
    output_fn=categorical_output_fn,
)

problem = Problem(
    env_make_fn=make,
    policy=policy,
    episode_length=200,
    num_rollouts=16,
    use_normalize_obs=True,
)

key, subkey = jax.random.split(key)
problem_state = problem.init(key)

key, subkey = jax.random.split(key)
solution = problem.sample(subkey)

print(f"Number of pararmeters: {sum(leaf.size for leaf in jax.tree.leaves(solution))}")


es = ES(
    population_size=population_size,
    solution=solution,
    optimizer=optax.adam(learning_rate=adam_lr),
)

params = es.default_params


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

plt.figure(figsize=(6, 3))
plt.plot(-metrics["best_fitness"])

plt.title("SNES")
plt.xlabel("Generations")
plt.ylabel("Fitness")

plt.grid(True)
plt.tight_layout()

plt.show()
