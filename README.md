# Continual control: evolution versus reinforcement learning


This is the code that accompanies our alife abstract


## Installing dependencies

We provide the library dependenices in the file [requirements.txt](requirements.txt).
You can create a virtual environment and install them using [uv](https://docs.astral.sh/uv/) with the following commands:

```
uv venv .
uv pip install -r requirements.txt
```

The package manager often installs different versions from the ones in the file, so you may need to make sure that you have the right verion manually. It is important to have the right versions for Brax- and jax-related libraries.


## Code overview
This repo contains the following directories:
* [methods](methods) contains the implementation of the methods we have benchmarked. (While we have employed existing libraries we have made internal changes to support curriculum learning and logging):
  * [brax](methods/RL) contains the implementation of PPO and goal-conditioned PPO (extending [Brax](https://github.com/google/brax/tree/main/brax))
  * [neuroevolution](methods/neuroevolution) contains the implementation of CMA-ES (this is a general framework for training direct encodings using [evosax](https://github.com/RobertTLange/evosax))

* [scripts](scripts) contains:
  * [train](scripts/train) scripts for rerunning traning. For each method we provide code for training in all tasks described in the paper, with the hyperparameters provided in a separate file 

## Training

To train methods on the control tasks you can run the script for evolution [scripts/train/evosax/train.py](scripts/train/evosax/train.py) and [scripts/train/rl/ppo/train.py](scripts/train/rl/ppo/train.py) for PPO


