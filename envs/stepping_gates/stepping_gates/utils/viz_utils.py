

from dgates import envs
import jax
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import numpy as onp
import jax.numpy as jnp


# ----- general figure configuration -----
cm = 1 / 2.54  # inches to cm
scale = 1
width, height = 3, 2
fig_size = (width / scale / cm, height / scale / cm)
params = {'legend.fontsize': 10,
          "figure.autolayout": True,
          'font.size': 10,
          "figure.figsize": fig_size}
plt.rcParams.update(params)



def viz_rollout(states, data, actions, env_config, save_dir):
    """ Visualize rollout in html.

    Parameters
    rollouts: (list of  pipeline). each item corresponds to a different rollout and each rollout is a pipeline state
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    total_reward = jnp.sum(data["reward"])
    num_timesteps = actions.shape[0]
    cum_reward = 0
    with open(save_dir + "/rollout_"+ str(total_reward) + ".txt", "w") as f:
        for step in range(num_timesteps):
            f.write("Step " + str(step) + "\n")

            obs = states.env_state.obs[step,...]
            f.write("Obs " + str(obs) + "\n")

            action = actions[step,...]
            f.write("Action " + str(action) + "\n")

            reward = data["reward"][step,...]
            f.write("Reward " + str(reward) + "\n")

            done = states.env_state.done[step,...]
            f.write("Done " + str(done) + "\n")

            cum_reward += reward

            if bool(done):
                f.write("Episode failed at step " + str(step))
                break

    print(cum_reward)
    print("check")




