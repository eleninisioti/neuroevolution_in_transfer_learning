# When Does Neuroevolution Outcompete Reinforcement Learning in Transfer Learning Tasks?

This is the code that accompanies our submission to GECCO 2025.
It enables running the experiments reported in the paper and producing visualisations.

## Set up

We have provided the library dependenices in [the pyproject.toml file](pyproject.toml). You can install them using [uv](https://docs.astral.sh/uv/) with the following command:

```
uv pip install
```


Our study considered two benchmarks: [ecorobot](https://github.com/eleninisioti/ecorobot) and [stepping gates](https://github.com/eleninisioti/stepping_gates). These are standalone github repos.
If you want to rerun training you will need to clone them under directory [envs](envs) using the following commands:


```
cd envs
git clone https://github.com/eleninisioti/ecorobot
git clone https://github.com/eleninisioti/stepping_gates
```

## Code organisation
This repo contains the following directories:
* [methods](methods) contains the implementation of the methods we have benchmarked. (While we have employed existing libraries we have made internal changes to support curriculum learning and logging):
  * [brax](methods/RL) contains the implementation of PPO and goal-conditioned PPO (extending [Brax](https://github.com/google/brax/tree/main/brax))
  * [tensorneat](methods/tensorneat) contains the implementation of NEAT and HyperNEAT (extending [tensorneat](https://github.com/EMI-Group/tensorneat))
  * [neuroevolution](methods/neuroevolution) contains the implementation of CMA-ES (this is a general framework for training direct encodings using [evosax](https://github.com/RobertTLange/evosax))

* [scripts](scripts) contains:
  * [train](scripts/train) scripts for rerunning traning. For each method we provide code for training in all tasks described in the paper, with the hyperparameters provided in a separate file 
  * [inspect](scripts/inspect) scripts for inspecting results for trained policies

## Inspecting results

We have provided all data produced by running our benchmarking in a zip file that you can download and extract in this project from [here]().
Alternatively, you can rerun training by calling the appropriate script from [scripts/train](scripts/train)(we have run all our experiments on a single NVIDIA RTX 6000 GPU. To provide an approximate estimate, a trial in stepping gates requires about 10 minutes of training and a trial in ecorobot about 30 minutes, with this of course differing across methods).

Experiments are saved under directory [experiments](experiments). An experiment contains the trained policies and information about evaluation.

To produce all paper visualisation you can run script [scripts/inspect/gecco_2025_visuals.py](scripts/inspect/gecco_2025_visuals.py). This will also print information regarding statistical significance.






If you would like to cite our study, please use:
```

@article{nisioti_when_2025,
	title = {When {Does} {Neuroevolution} {Outcompete} {Reinforcement} {Learning}  in {Transfer} {Learning} {Tasks}?},
	language = {en},
	author = {Nisioti, Eleni and Pedersen, Joachim Winther and Plantec, Erwan and Montero, Milton L and Risi, Sebastian},
	year = {2025},
}
````
