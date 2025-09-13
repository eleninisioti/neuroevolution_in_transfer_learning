
# ------ total training time -------
train_gens = {"n_parity": 5000,
              "n_parity_only_n": 5000,
                   "simple_alu": 10000,
                   "maze_with_stepping_stones": 2000,
                   "locomotion_with_obstacles": 500,
                   "locomotion": 2000,
                   "deceptive_maze_easy": 100}

# ------ PPO hyperparams for brax implementation -------
hyperparams = {
    "n_parity": {
                        "mut_node": 0.1,
                        "mut_conn": 0.1,
                        "max_nodes": 100,
                        "node_add": 0.1,
                        "conn_add": 0.2,
                        "node_delete": 0.1,
                        "conn_delete": 0.2
                 },
    "n_parity_only_n": {
                        "mut_node": 0.1,
                        "mut_conn": 0.1,
                        "max_nodes": 100,
                        "node_add": 0.1,
                        "conn_add": 0.2,
                        "node_delete": 0.1,
                        "conn_delete": 0.2
                 },
       "simple_alu": {
                            "mut_node": 0.1,
                            "mut_conn": 0.1,
                            "max_nodes": 100,
                            "node_add": 0.1,
                            "conn_add": 0.2,
                            "node_delete": 0.1,
                            "conn_delete": 0.2
                     },
       "maze_with_stepping_stones": {
                            "mut_node": 0.1,
                            "mut_conn": 0.1,
                            "max_nodes": 100,
                            "node_add": 0.1,
                            "conn_add": 0.2,
                            "node_delete": 0.1,
                            "conn_delete": 0.2
                     },
       "locomotion_with_obstacles": {
                            "mut_node": 0.1,
                            "mut_conn": 0.1,
                            "max_nodes": 100,
                            "node_add": 0.1,
                            "conn_add": 0.2,
                            "node_delete": 0.1,
                            "conn_delete": 0.2
                     },
       "locomotion": {
                            "mut_node": 0.1,
                            "mut_conn": 0.1,
                            "max_nodes": 100,
                            "node_add": 0.1,
                            "conn_add": 0.2,
                            "node_delete": 0.1,
                            "conn_delete": 0.2
                     },            
       "deceptive_maze_easy": {
                            "mut_node": 0.1,
                            "mut_conn": 0.1,
                            "max_nodes": 100,
                            "node_add": 0.1,
                            "conn_add": 0.2,
                            "node_delete": 0.1,
                            "conn_delete": 0.2
                     }
}



