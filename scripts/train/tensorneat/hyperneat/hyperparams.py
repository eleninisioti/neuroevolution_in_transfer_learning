
# ------ total training time -------
train_gens = {"n_parity_only_n": 5000,
                   "simple_alu": 10000}

# ------ PPO hyperparams for brax implementation -------
hyperparams = {
    "n_parity_only_n": {"num_hidden_neurons_per_layer": 16,
                        "num_hidden_layers": 1,
                        "mut_node": 0.1,
                        "mut_conn": 0.1,
                        "max_nodes": 100,
                        "node_add": 0.1,
                        "conn_add": 0.2,
                        "cppn_max_nodes": 100,
                        "cppn_max_conns": 1000,
                        "node_delete": 0.1,
                        "conn_delete": 0.2
                 },
   "simple_alu": {"num_hidden_neurons_per_layer": 16,
                        "num_hidden_layers": 1,
                        "mut_node": 0.1,
                        "mut_conn": 0.1,
                        "max_nodes": 100,
                        "node_add": 0.1,
                        "conn_add": 0.2,
                        "cppn_max_nodes": 100,
                        "cppn_max_conns": 1000,
                        "node_delete": 0.1,
                        "conn_delete": 0.2
                 },
}



