import numpy as np

from .func_fit import FuncFit
import jax.numpy as jnp
import jax
import os
import imageio
import matplotlib.pyplot as plt
class NparityCurric(FuncFit):

    def __init__(self, n_input, error_method: str = "mse"):
        self.n_input = n_input
        self.observation_size = self.n_input
        self.action_size = 1
        self.fitness_target = -(1/(2 ** self.n_input)-0.001)
        self.num_tasks = n_input
        self.curriculum = True
        self.episode_length =2 ** self.n_input
        super().__init__(error_method)

    def preprocess_action(self, action):
        return jnp.where(action > 0.5, 1.0, 0.0)

    def setup(self, state):
        state = super().setup(state)
        state = state.register(n_active_inputs=2)
        return state


    def get_inputs(self, state):
        active_inputs = state.n_active_inputs
        nums = jnp.arange(2 ** self.n_input)
        # Convert each integer to its binary representation with n bits
        combinations = jnp.array([(nums >> i) & 1 for i in range(self.n_input- 1, -1, -1)]).T


        def deactivate_inputs(inputs):
            return jnp.where(jnp.arange(self.n_input) >= active_inputs, -1, inputs)

        combinations = jax.vmap(deactivate_inputs)(combinations)
        # append zeros to each combination
        return combinations.astype(jnp.float32)

    def get_targets(self, state):
        inputs = self.get_inputs(state)
        active_inputs = state.n_active_inputs

        def nparity( obs):
            active_obs = jnp.where(jnp.arange(self.n_input) >= active_inputs, 0, obs)
            targets = jnp.array([(jnp.sum(active_obs) % 2)])
            return targets

        outputs = jax.vmap(nparity)(inputs)
        return outputs.astype(jnp.float32)

    @property
    def input_shape(self):
        return jnp.arange(2 ** self.n_input), self.n_input

    @property
    def output_shape(self):
        return jnp.arange(2 ** self.n_input), 1

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
        secs_per_step = 2
        with imageio.get_writer(gif_path, mode='I', duration=len(data)*secs_per_step) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))

        return gif_path
