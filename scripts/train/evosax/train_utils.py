
import functools
import os
import envs
import pickle
from scripts.train.rl.ppo.hyperparams import hyperparams
from scripts.train.base.experiment import Experiment
from functools import partial
import numpy as onp
import jax.numpy as jnp
import jax
from scripts.train.base.visuals import viz_histogram, viz_heatmap
from scripts.train.rl.ppo.hyperparams import hyperparams
from stepping_gates import envs as stepping_gates_envs
from ecorobot import envs as ecorobot_envs
from methods.evosax_wrapper.base.tasks.rl import EcorobotTask
from methods.evosax_wrapper.direct_encodings.model import make_model
import methods.evosax_wrapper.evosax
from methods.evosax_wrapper.base.training.evolution import EvosaxTrainer
from methods.evosax_wrapper.base.training.logging  import Logger
import equinox as eqx
import evosax
from methods.evosax_wrapper.base.tasks.rl import GatesTask
import wandb


def _unpmap(v):
  return jax.tree_util.tree_map(lambda x: x[0], v)

class EvosaxExperiment(Experiment):

    def __init__(self, env_config, model_config, exp_config, optimizer_config):
        super().__init__(env_config, model_config, exp_config, optimizer_config)
        

    def setup_trial_keys(self):
        key = jax.random.PRNGKey(self.config["exp_config"]["trial_seed"])

        self.model_key, self.train_key = jax.random.split(key, 2)
        
    def init_model(self):
      self.model = make_model(self.config, self.model_key)

    def cleanup(self):
        pass
    
    
    
    def setup_stepping_gates_env(self):
        
        self.env = stepping_gates_envs.get_environment(env_name=self.config["env_config"]["env_name"],
                                                      **self.config["env_config"]["env_params"])
        
        self.config["env_config"]["action_size"] = self.env.action_size
        self.config["env_config"]["observation_size"] = self.env.observation_size
        self.config["env_config"]["episode_length"] = self.env.episode_length
        self.config["env_config"]["num_tasks"] = self.env.num_tasks
        
    def setup_ecorobot_env(self):
        self.env = ecorobot_envs.get_environment(env_name=self.config["env_config"]["env_name"],
                                                      **self.config["env_config"]["env_params"])
        
        self.config["env_config"]["action_size"] = self.env.action_size
        self.config["env_config"]["observation_size"] = self.env.observation_size
        self.config["env_config"]["episode_length"] = self.env.episode_length
        self.config["env_config"]["num_tasks"] = self.env.num_tasks

    





    def metrics_fn(self, log_info,  data,task_params, num_nodes, num_edges):


        def callback(log_info, task_params, data, num_nodes, num_edges):
            log_info = {
                "current_best_fitness": onp.max(onp.array(data["fitness"])),
                "generation": log_info.gen_counter,
                "current_task": task_params,
                #"diversity": log_info.diversity,
                #"navigability": log_info.navig,
                #"navigability_online": log_info.navig_online,
                #"robustness_fitness": log_info.robustness_fitness,
                #"robustness": log_info.robustness,
                "num_nodes": num_nodes,
                "num_edges": num_edges

            }

            wandb.log(log_info)
            
            for key, value in log_info.items():
                print(key, value)

        jax.debug.callback(callback, log_info, task_params, data, num_nodes, num_edges)

    def eval_task(self, best_member, tasks, gens, final_policy=False):

        policy_params = self.params_shaper.reshape_single(best_member)

        policy = eqx.combine(policy_params, self.statics)


        init_policy_state, _ = policy.initialize(jax.random.PRNGKey(0))



        act_fn = partial(policy, key=self.model_key, state=init_policy_state, )


        super().run_eval(act_fn, tasks, final_policy=final_policy, gens=gens)

   
    def get_final_policy(self):
        policy_params = self.params_shaper.reshape_single(self.final_state["params"])

        policy = eqx.combine(policy_params, self.statics)

        init_policy_state, dev_states = policy.initialize(jax.random.PRNGKey(0))
        dev_steps = self.config["model_config"]["model_params"]["max_dev_steps"] + 2
        data = jax.tree_map(lambda x: x[dev_steps, ...], dev_states)
        return data.weights

    def train_trial(self):

        def data_fn(data: dict):
            return {}

        logger = Logger(True,
                        metrics_fn=self.metrics_fn,
                        ckpt_freq=100,
                        aim_freq=10,
                        ckpt_dir=self.config["exp_config"]["trial_dir"] + "/data/train")

        fitness_shaper = evosax.FitnessShaper(maximize=True,
                                              centered_rank=False)

        phenotype_size = (self.model.max_nodes, self.model.max_nodes)
        params, statics = eqx.partition(self.model, eqx.is_array)
        self.statics = statics
        self.params_shaper = evosax.ParameterReshaper(params)

    
        """
        self.env = GatesTask(statics,

                        env=self.config["env_config"]["env_name"],
                        max_steps= self.config["env_config"]["episode_length"],
                        data_fn=data_fn, env_kwargs={**self.config["env_config"]["env_params"]})
        """
        
        self.env = EcorobotTask(statics=self.statics,
                                env=self.config["env_config"]["env_name"],
                                max_steps=1000,
                                data_fn=data_fn,
                                env_kwargs={**self.config["env_config"]["env_params"]})
       



        trainer = EvosaxTrainer(train_steps=self.config["optimizer_config"]["optimizer_params"]["generations"],
                                task=self.env,
                                save_params_fn=self.save_params,
                                strategy=self.config["optimizer_config"]["optimizer_params"]["strategy"],
                                params_shaper=self.params_shaper,
                                popsize=self.config["optimizer_config"]["optimizer_params"]["popsize"],
                                fitness_shaper=fitness_shaper,
                                num_tasks = self.env.num_tasks,
                                reward_for_solved=self.env.reward_for_solved,
                                # sigma_init = 0.01,
                                es_kws={
                                        },
                                logger=logger,
                                progress_bar=False,
                                n_devices=1,
                                eval_reps=2)

        final_info = trainer.init_and_train_(self.train_key)

        self.final_state = {"params": final_info.best_member}
        
        
        
    def save_params(self, training_state):

        def callback(info):
            current_gen, current_task, state, interm_policies, best_indiv = info
            last_dev_step = 1
            best_member = jax.tree_map(lambda x: x[best_indiv, ...], state)

            file_path = self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/params_task_" + str(current_task-1) + ".pkl"
            if not os.path.exists(file_path):

                with open(file_path, "wb") as f:
                    pickle.dump( (current_gen,best_member), f)


            interm_policies = jax.tree_map(lambda x: x[best_indiv,0, ...], interm_policies)


            file_path = self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/policy_states_task_" + str(
                current_task - 1) + ".pkl"
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    pickle.dump(interm_policies, f)

        jax.debug.callback(callback, training_state)

    def save_training_info(self):


        #TODO: here we need to load from latest generation

        #last_gen = self.config["optimizer_config"]["optimizer_params"]["generations"]
        #with open(self.config["exp_config"]["trial_dir"] + "/data/train/all_info/gen_" + str(last_gen) + "/dev_" +str(self.config["model_config"]["model_params"]["max_dev_steps"]+2) +".pkl", "rb") as f:
        #    policy_state = pickle.load(f)
        policy_state = self.final_state["params"]

        checkpoint_policy_states = []
        for task in range(self.config["env_config"]["num_tasks"]):

            try:

                with open(self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/params_task_" + str(
                        task) + ".pkl","rb") as f:
                    data = pickle.load(f)
                    checkpoint_policy_states.append(data)
            except FileNotFoundError:
                continue

        # save final policy matrix
        self.training_info = {"policy_network": {"final": policy_state,
                                                 "checkpoints": checkpoint_policy_states}}

        super().save_training_info()