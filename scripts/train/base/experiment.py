from datetime import datetime
import os
import jax.random
import yaml
import envs
import copy
import envs
import pickle
import numpy as onp
import subprocess
import wandb
import matplotlib.pyplot as plt
from stepping_gates import envs as stepping_gates_envs
from brax import envs as brax_envs
import shutil
import equinox as eqx
import  gymnasium as gym
from scripts.train.base.task import Task
from scripts.train.base.visuals import viz_histogram



class Experiment:

    def __init__(self, env_config={}, model_config={}, exp_config={}, optimizer_config={}):
        """ Base class for tracking an experiment
        """
        self.config = {"env_config": env_config,
                       "model_config": model_config,
                       "optimizer_config": optimizer_config,
                       "exp_config": exp_config}
        
    def setup(self):

        self.env_alias = self.config["env_config"]["env_type"] + "_" + self.config["env_config"][
            "env_name"] + "_" + "_".join(
            f"{key}_{value}" for key, value in self.config["env_config"]["env_params"].items())
        self.model_alias = self.config["model_config"]["network_type"] + "_" + "_".join(
            f"{key}_{value}" for key, value in self.config["model_config"]["model_params"].items())

        self.opt_alias = self.config["optimizer_config"]["optimizer_type"] + "_" + self.config["optimizer_config"][
            "optimizer_name"] + "/" + "_".join(
            f"{key}_{value}" for key, value in self.config["optimizer_config"]["optimizer_params"].items())

        project_dir = "projects/benchmarking/" + datetime.today().strftime(
            '%Y_%m_%d') + "/" + self.env_alias + "/" + self.opt_alias + "/" + self.model_alias 

        print("Saving project under " + project_dir)

        self.config["exp_config"]["project_dir"] = project_dir

        

        if not os.path.exists(project_dir):
            os.makedirs(project_dir, exist_ok=True)

        # make project directories
        for trial in range(self.config["exp_config"]["num_trials"]):
            os.makedirs(project_dir + "/trial_" + str(trial) + "/data/train/checkpoints", exist_ok=True)
            os.makedirs(project_dir + "/trial_" + str(trial) + "/data/eval/trajs", exist_ok=True)
            os.makedirs(project_dir + "/trial_" + str(trial) + "/visuals/eval/trajs", exist_ok=True)
            os.makedirs(project_dir + "/trial_" + str(trial) + "/visuals/train/policy", exist_ok=True)
            os.makedirs(project_dir + "/trial_" + str(trial) + "/visuals/eval/network_activations", exist_ok=True)

        
        self.setup_env()
        
        with open(self.config["exp_config"]["project_dir"] + "/config.yaml", "w") as f:
            yaml.dump(self.config, f)

   

    def setup_env(self):

        if self.config["env_config"]["env_type"] == "stepping_gates":
            self.setup_stepping_gates_env()
            
        elif self.config["env_config"]["env_type"] == "brax":
            self.setup_brax_env()
            
        elif self.config["env_config"]["env_type"] == "ecorobot":
            self.setup_ecorobot_env()
            
        elif self.config["env_config"]["env_type"] == "gymnax":
            self.setup_gymnax_env()

        self.task = Task(self.env, self.config)


    def setup_trial_keys(self):
        pass

    def init_run(self):
        self.init_model()

    def setup_trial(self, trial):
        self.config["exp_config"]["trial_dir"] = self.config["exp_config"]["project_dir"] + "/trial_" + str(trial)
        self.config["exp_config"]["experiment"] = self.model_alias + "_" + self.opt_alias + "_trial_" + str(trial)
        self.config["exp_config"]["trial_seed"] = trial

        # start logging
        wandb.init(
            project=self.env_alias,
            name=self.opt_alias,
            tags =  "/trial_" + str(trial),
            config=self.config,
            reinit=True
        )        
        self.setup_trial_keys()

        self.init_model()


    def cleanup_trial(self):
        #er_run.close()
        wandb.finish()
        
        print("Experiment data saved under ", self.config["exp_config"]["project_dir"])
        
    def train_trial(self):
        pass
    

    def run(self):
        self.setup()

        for trial in range(self.config["exp_config"]["num_trials"]):
            self.setup_trial(trial)

            self.train_trial()
            
            self.save_training_info()

            self.eval_trial()

            self.cleanup_trial()

        self.cleanup()

    def viz_final_policy(self):

        weights = self.get_final_policy()


        viz_heatmap(weights,
                    filename=self.config["exp_config"]["trial_dir"] + "/visuals/policy/heatmap")
        """
        viz_policy_network(weights=self.training_info["policy_network"]["weights"],
                           n_input=self.config["env_config"]["observation_size"],
                           n_output=self.config["env_config"]["action_size"],
                           filename=self.config["exp_config"]["trial_dir"] + "/visuals/policy/network",
                           network_type=self.config["model_config"]["network_type"])
        """

        # viz_heatmap(self.training_info["policy_network"]["bias"], self.config["exp_config"]["trial_dir"], "/policy_bias_heatmap")
        # viz_policy_network(self.training_info["policy_network"]["weights"], self.training_info["policy_network"]["bias"], self.config["exp_config"]["trial_dir"], "policy_network" )

    def viz_network_activation(self):
        for step in range(len(self.eval_info["first_trial_actions"])):
            actions = onp.array(self.eval_info["first_trial_actions"][step])
            obs = onp.array(self.eval_info["first_trial_obs"][step])
            viz_policy_network(weights=self.training_info["policy_network"]["weights"],
                               obs=obs,
                               actions=actions,
                               n_input=self.config["env_config"]["observation_size"],
                               n_output=self.config["env_config"]["action_size"],
                               filename=self.config["exp_config"][
                                            "trial_dir"] + "/visuals/eval/network_activations/step" + str(step))

    def viz_eval(self, eval_info):
        # plot rewards for each task
        task_success = []
        for task, values in eval_info.items():
            image_file = viz_histogram(values["rewards"], filename=self.config["exp_config"][
                                                                       "trial_dir"] + "/visuals/eval/rewards_task_" + str(
                task))
            """
            aim_image = Image(
                image_file,
                format='png',
                optimize=True,
                quality=50
            )
            self.logger_run.track(aim_image, name="rewards_task" + str(task))
            """
            task_success.append(onp.mean(values["success"]))

        image_file = viz_histogram(task_success, filename=self.config["exp_config"][
                                                              "trial_dir"] + "/visuals/eval/success")
        """
        aim_image = Image(
            image_file,
            format='png',
            optimize=True,
            quality=50
        )
        self.logger_run.track(aim_image, name="success")
        """

    def save_training_info(self):
            
        final_state = {k: v for k, v in self.final_state.items() if k != 'inference_fn'}

        with open(self.config["exp_config"]["trial_dir"] + "/data/train/final_state.pkl", "wb") as f:
            pickle.dump(final_state, f)
            
            
        with open(self.config["exp_config"]["trial_dir"] + "/data/train/final_policy.pkl", "wb") as f:
            pickle.dump(self.training_info, f)

    def get_params(self, f):
        return pickle.load(f)

    def eval_trial(self):
        top_dir = self.config["exp_config"]["trial_dir"] + "/data/train/checkpoints/"
        param_files = os.listdir(top_dir)
        for task in range(self.task.num_tasks):
            params_file = "params_task_" + str(task) + ".pkl"

            if params_file in param_files:
                with open(os.path.join(top_dir, params_file), "rb") as f:
                    
                    gens, params = self.get_params(f)
                    self.eval_task(params, gens=gens, tasks=[task])

            else:
                print("Task " + str(task) + " has not been solved so not running evaluation for it")

        params = self.final_state["params"]
        self.eval_task(params, gens=-1,tasks=range(self.task.num_tasks), final_policy=True)

        with open(self.config["exp_config"]["trial_dir"] + "/data/eval/info.yml", "w") as f:
            yaml.dump(self.task.eval_info, f)

        self.viz_eval(self.task.eval_info)

    def run_eval(self, act_fn, tasks, gens, final_policy=False):

        self.task.run_eval(act_fn, self.config["exp_config"]["trial_dir"] + "/visuals/eval/trajs", tasks, gens, final_policy)

    