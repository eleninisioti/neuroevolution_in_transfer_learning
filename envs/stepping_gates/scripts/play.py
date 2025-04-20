import sys
sys.path.append(".")
sys.path.append('envs/stepping_gates')
import jax.random
import jax.numpy as jnp
from stepping_gates import envs

def play_stepping_gates(task, curriculum, episode_type):


    env = envs.get_environment(task, curriculum=curriculum, episode_type=episode_type)

    key = jax.random.PRNGKey(0)
    state = env.reset(key, env_params=jnp.array([1]))
    done = False

    cum_reward = 0
    while (True):

        action = jnp.array([0.2])
        print(state.obs)
        state = env.step(state, action)
        print(action, state.info["label"], state.reward)
        print(state)
        # env.render(state)
        cum_reward += float(state.reward)
        print("Reward so far" + str(cum_reward))

        if state.done:
            break






if __name__ == "__main__":
    #play_stepping_gates(task="n_parity", curriculum=False, episode_type="all_steps")

    #play_stepping_gates(task="n_parity", curriculum=True, episode_type="all_steps")

    #play_stepping_gates(task="n_parity_only_n", curriculum=False, episode_type="all_steps")
    play_stepping_gates(task="n_parity_only_n", curriculum=False, episode_type="one-step")


    #play_stepping_gates(task="simple_alu", curriculum=False, episode_type="all_steps")
    #play_stepping_gates(task="simple_alu", curriculum=True, episode_type="all_steps")
    quit()

    play_stepping_gates(task="simple_alu_only_n", curriculum=False, episode_type="all_steps")
    play_stepping_gates(task="n_parity_only_n", curriculum=False, episode_type="all_steps")




