import jax
from jax import vmap, numpy as jnp
import jax.numpy as jnp

from ..base import BaseProblem
from tensorneat.common import State


class FuncFit(BaseProblem):
    jitable = True

    def __init__(self, error_method: str = "mse"):
        super().__init__()

        assert error_method in {"mse", "rmse", "mae", "mape"}
        self.error_method = error_method

    def setup(self, state: State = State()):
        return state

    def evaluate(self, state, randkey, act_func, params):
        inputs = self.get_inputs(state)


        predict = vmap(act_func, in_axes=(None, None, 0))(
            state, params, inputs
        )
        predict = self.preprocess_action(predict)


        #predict = self.preprocess_action(predict)
        targets = self.get_targets(state)
        if self.error_method == "mse":
            loss = jnp.mean((predict - targets) ** 2)

        elif self.error_method == "rmse":
            loss = jnp.sqrt(jnp.mean((predict - targets) ** 2))

        elif self.error_method == "mae":
            loss = jnp.mean(jnp.abs(predict - targets))

        elif self.error_method == "mape":
            loss = jnp.mean(jnp.abs((predict - targets) / targets))

        else:
            raise NotImplementedError

        return -loss

    def show(self, state, randkey, act_func, params, *args, **kwargs):
        inputs = self.get_inputs(state)
        targets = self.get_targets(state)


        predict = vmap(act_func, in_axes=(None, None, 0))(
            state, params, inputs
        )

        predict = self.preprocess_action(predict)

        inputs, target, predict = jax.device_get([inputs, targets, predict])
        fitness = self.evaluate(state, randkey, act_func, params)

        #processed_predict = jnp.where(predict > 0.5, 1, 0)
        success = jnp.where(predict == target, 1, 0)

        loss = -fitness

        msg = ""
        for i in range(inputs.shape[0]):
            msg += f"input: {inputs[i]}, target: {target[i]}, predict: {predict[i]}\n"
        msg += f"loss: {loss}\n"
        print(msg)
        return inputs, target, predict, fitness, success

    @property
    def inputs(self):
        raise NotImplementedError

    @property
    def targets(self):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError
