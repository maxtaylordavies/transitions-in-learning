import jax

from transitions_in_learning.environment import make, EnvState


def print_state(state: EnvState):
    print("Time:", state.time)
    print("Energy:", state.energy)
    print(
        "Current Resource:",
        ["None", "Cardboard", "Food", "Poison"][int(state.current_resource)],
    )
    print("Mouth Open:", state.mouth_open)
    print("Pain Level:", state.pain_level)
    print("Pain Delta Buffer:", state.pain_delta_buffer)
    print("Pain Pointer:", state.pain_ptr)


key = jax.random.PRNGKey(0)
env, env_params = make()

_, state = env.reset(key, env_params)
print("Initial State:")
print_state(state)
print("-" * 20)

for _ in range(10):
    key, subkey = jax.random.split(key)

    # get action from user input
    action = int(input("Enter action (0: NO_OP, 1: OPEN_MOUTH, 2: CLOSE_MOUTH): "))

    obs, state, reward, done, info = env.step(subkey, state, action, env_params)

    print("New State:")
    print_state(state)
    print("Reward:", reward)
    print("Done:", done)
    print("-" * 20)
