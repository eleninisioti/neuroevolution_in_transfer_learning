import numpy as np

from .func_fit import FuncFit
import jax.numpy as jnp
import jax
import os
import imageio
import matplotlib.pyplot as plt
from flax import struct

@struct.dataclass
class State:
    obs: jnp.ndarray
    reward: jnp.ndarray
    info: jnp.ndarray
    done: jnp.ndarray


class Nparity(FuncFit):

    def __init__(self, n_input, error_method: str = "mse"):
        self.n_input = n_input
        self.observation_size = self.n_input
        self.action_size = 1
        self.fitness_target = -(1/(2 ** self.n_input)-0.001)
        self.curriculum = False
        self.num_tasks = 1
        self.task_params = jnp.array([6])

        self.episode_length = 2 ** self.n_input
        super().__init__(error_method)

    def setup(self, state):
        state = super().setup(state)
        state = state.register(n_active_inputs=self.n_input)
        return state

    def get_inputs(self, state={}):
        nums = jnp.arange(2 ** self.n_input)
        # Convert each integer to its binary representation with n bits
        combinations = jnp.array([(nums >> i) & 1 for i in range(self.n_input - 1, -1, -1)]).T
        return combinations.astype(jnp.float32)


    def preprocess_action(self, action):
        return jnp.where(action > 0.5, 1.0, 0.0)

    def get_targets(self, state):
        inputs = self.get_inputs(state)
        def nparity( obs):
            return jnp.array([(jnp.sum(obs) % 2)])

        outputs = jax.vmap(nparity)(inputs)

        return outputs.astype(jnp.float32)


    def reset(self, key, env_params):
        inputs = self.get_inputs()
        obs= jax.random.choice(key, inputs)
        label = jnp.array([(jnp.sum(obs) % 2)])

        state = State(obs=obs, reward=0.0, info={"label": label}, done=False)
        return state

    def step(self, state, action):
        action = self.preprocess_action(action)
        label = jnp.array([(jnp.sum(state.obs) % 2)])
        reward = jnp.where(action==label, 1, 0)[0]
        return state.replace(reward=reward, info={"label": label}, done=True)


    @property
    def input_shape(self):
        return jnp.arange(2 ** self.n_input), self.n_input

    @property
    def output_shape(self):
        return jnp.arange(2 ** self.n_input), 1

    def show_rollout(self, data, save_dir, filename):
        output_dir = save_dir 
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
        secs_per_step = 2
        with imageio.get_writer(gif_path, mode='I', duration=len(data)*secs_per_step) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))

        return gif_path
