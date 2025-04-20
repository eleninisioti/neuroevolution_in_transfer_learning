from stepping_gates.envs import simple_alu_only_n
from stepping_gates.envs import simple_alu
from stepping_gates.envs import n_parity
from stepping_gates.envs import n_parity_only_n
from stepping_gates.envs.base import Circuit

_envs = {"simple_alu": simple_alu.SimpleALU,
         "n_parity": n_parity.NParity,
         "simple_alu_only_n": simple_alu_only_n.SimpleALU,
         "n_parity_only_n": n_parity_only_n.NParity,
         }

def get_environment(env_name, **kwargs) -> Circuit:
  """Returns an environment from the environment registry.

  Args:
    env_name: environment name string
    **kwargs: keyword arguments that get passed to the Env class constructor

  Returns:
    env: an environment
  """
  return _envs[env_name](**kwargs)


def create(
    env_name: str,
    **kwargs,
) -> Circuit:
  """Creates an environment from the registry.

  Args:
    env_name: environment name string
  Returns:
    env: an environment
  """
  env = _envs[env_name](**kwargs)

  return env

