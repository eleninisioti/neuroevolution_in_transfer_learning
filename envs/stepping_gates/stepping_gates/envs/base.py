
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from flax import struct
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

@struct.dataclass
class State:
    obs: jnp.ndarray
    info: jnp.ndarray
    done: jnp.ndarray
    reward: jnp.ndarray
    metrics: jnp.ndarray


class Circuit:
    def __init__(self):
        super().__init__()


    def get_obs_size(self, task):

        return self.observation_space.shape[0]

    def get_action_size(self, task):

        return self.action_space.shape[0]


    def reset(self, key):
        # Reset the state to the initial state
        pass

    def step(self, state, action):
        # Apply action to the environment and compute the next state and reward
        pass

    def render(self, mode='human'):
        # Render the environment to the screen (optional)
        pass

    def preprocess_action(self, action):
        " Makes the action binary"
        return jnp.where(jnp.greater_equal(jax.nn.sigmoid(action), 0.5), jnp.ones(action.shape), jnp.zeros(action.shape))

    @property
    def action_size(self):
        return self.action_space.shape[0]

    @property
    def observation_size(self):
        return self.observation_space.shape[0]


    def show_rollout(self, data, output_dir, filename):
        frame_paths = []
        if not os.path.exists(output_dir + "/" + filename):
            os.makedirs(output_dir + "/" + filename)
        for idx, frame_data in enumerate(data):
            fig, ax = plt.subplots(figsize=(12, 2))  # Adjust the size as needed
            ax.axis('off')  # Turn off axes

            # Display the data as a table
            table = ax.table(cellText=frame_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(12)

            # Save the frame

            frame_path = os.path.join(output_dir + "/" + filename, "step_" + str(idx) +".png")
            plt.tight_layout()
            plt.savefig(frame_path, bbox_inches='tight', dpi=150)
            frame_paths.append(frame_path)
            plt.close()

        # Create a GIF from the frames
        gif_path = output_dir + "/" + filename+ ".gif"
        secs_per_step = 5
        with imageio.get_writer(gif_path, mode='I', duration=len(data)*secs_per_step) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))

        return gif_path
