
# TODO: The merge dicts for hyperparam sweeps is a little annoying, maybe program a better way

from util import merge_dict

sweep_configs = {}

# ------------------ Experiment Sweeps ------------------

# BASELINE - Combine with an experiment-specific dict

exp_baseline = {
    'method': 'grid',
    'metric': {'goal': 'minimize', 'name': 'loss'},
    'parameters': 
    {
        'rand_seed': {'values': [5632, 6145, 5123, 4626, 4117, 1558, 9241, 5155, 6193, 5685, 
                                 3646, 4673, 5701, 8262, 3153, 3670, 5208, 8292, 1128, 618, 
                                 2159, 4720, 6258, 9854, 8833, 141, 2195, 5784, 5787, 670, 
                                 6305, 8866, 3241, 8881, 2237, 8382, 7364, 713, 4809, 7887, 
                                 5329, 5341, 2270, 9443, 8935, 3314, 4339, 9468, 2815, 4354, 
                                 2313, 3364, 9002, 6444, 1838, 3889, 4403, 5435, 9531, 3901, 
                                 1854, 4924, 324, 325, 5956, 1861, 8527, 3927, 3419, 860, 
                                 9570, 8549, 361, 8047, 3440, 3952, 6000, 8574, 7553, 6536, 
                                 3465, 1429, 6551, 6039, 412, 2975, 8100, 7589, 2983, 9127, 
                                 3502, 7093, 449, 6596, 6605, 3541, 7125, 6625, 9708, 6133]
        },
        'print_interval': {'value': 50},
        'device': {'value': 'lightning.qubit'},
     }
}

# SMALL ISING EXPERIMENT

small_ising_exp = {
    'parameters': 
    {
        'num_qubits': {'value': 3},
        'problem': {'value': 'transverse_ising'},
        'model': {'value': 'rand_layers'},
        'num_layers': {'value': 3},
        'num_params': {'value': 10},
        'ratio_imprim': {'value': 0.3},
        'steps': {'value': 500},
     }
}
small_ising_exp_baseline = merge_dict(exp_baseline, small_ising_exp)

small_ising_spsa_exp = {
    'name': 'Small Ising SPSA Random Experiments',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': 'spsa'},
        'est_shots': {'value': 1},
        'stddev': {'value': 0.2},
        'alpha': {'value': 0.602},
        'gamma': {'value': 0.101},
     }
}
sweep_configs["small_ising_spsa_exp"] = merge_dict(small_ising_exp_baseline, small_ising_spsa_exp)

small_ising_adam_spsa_exp = {
    'name': 'Small Ising Adam SPSA Random Experiments',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': 'adamspsa'},
        'est_shots': {'value': 1},
        'stddev': {'value': 0.2},
        'alpha': {'value': 0.602},
        'gamma': {'value': 0.101},
        'learning_rate': {'value': 0.1},
        'beta': {'value': 0.99},
        'lmd': {'value': 0.42},
        'zeta': {'value': 0.99},
     }
}
sweep_configs["small_ising_adam_spsa_exp"] = merge_dict(small_ising_exp_baseline, small_ising_adam_spsa_exp)

small_ising_qnspsa_exp = {
    'name': 'Small Ising QNSPSA Random Experiments',
    'parameters': 
    {
        'device': {'value': 'default.qubit'},
        'interface': {'value': 'numpy'},
        'optimizer': {'value': 'pl_qnspsa'},
        'est_shots': {'value': 1},
        'learning_rate': {'value': 0.01},
        'stddev': {'value': 0.01},
        'metric_reg': {'value': 0.001},
     }
}
sweep_configs["small_ising_qnspsa_exp"] = merge_dict(small_ising_exp_baseline, small_ising_qnspsa_exp)

small_ising_spsa2_exp = {
    'name': 'Small Ising 2-SPSA Random Experiments',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': '2spsa'},
        'est_shots': {'value': 1},
        'learning_rate': {'value': 0.1},
        'stddev': {'value': 0.01},
        'metric_reg': {'value': 0.001},
     }
}
sweep_configs["small_ising_spsa2_exp"] = merge_dict(small_ising_exp_baseline, small_ising_spsa2_exp)

small_ising_xnes_exp = {
    'name': 'Small Ising xNES Random Experiments',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': 'xnes'},
        'est_shots': {'value': 2},
        'nu_sigma': {'value': 0.001},
        'nu_b': {'value': 0.001},
        'nu_mu': {'value': 0.1},
        'stddev': {'value': 0.1},
    }
}
sweep_configs["small_ising_xnes_exp"] = merge_dict(small_ising_exp_baseline, small_ising_xnes_exp)

small_ising_snes_exp = {
    'name': 'Small Ising sNES Random Experiments',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': 'snes'},
        'est_shots': {'value': 2},
        'nu_sigma': {'value': 0.01},
        'nu_mu': {'value': 0.1},
        'stddev': {'value': 0.1},
    }
}
sweep_configs["small_ising_snes_exp"] = merge_dict(small_ising_exp_baseline, small_ising_snes_exp)

small_ising_ges_exp = {
    'name': 'Small Ising GES Random Experiments',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': 'ges'},
        'est_shots': {'value': 1},
        'learning_rate': {'value': 0.1},
        'explore_tradeoff': {'value': 0.5},
        'grad_scale': {'value': 2},
        'stddev': {'value': 0.01},
        'grad_memory': {'value': 10},
    }
}
sweep_configs["small_ising_ges_exp"] = merge_dict(small_ising_exp_baseline, small_ising_ges_exp)

# RANDOMIZED HAMILTONIAN EXPERIMENT

random_hamiltonian_exp = {
    'parameters': 
    {
        'num_qubits': {'value': 10},
        'problem': {'value': 'randomized_hamiltonian'},
        'num_random_singles' : {'value': 10},
        'num_random_doubles' : {'value': 20},
        'model': {'value': 'rand_layers'},
        'num_layers': {'value': 3},
        'num_params': {'value': 10},
        'ratio_imprim': {'value': 0.3},
        'steps': {'value': 500},
     }
}
random_hamiltonian_exp_baseline = merge_dict(exp_baseline, random_hamiltonian_exp)

random_hamiltonian_spsa_exp = {
    'name': 'Random Hamiltonian SPSA Random Experiments',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': 'spsa'},
        'est_shots': {'value': 1},
        'stddev': {'value': 0.2},
        'alpha': {'value': 0.602},
        'gamma': {'value': 0.101},
     }
}
sweep_configs["random_hamiltonian_spsa_exp"] = merge_dict(random_hamiltonian_exp_baseline, random_hamiltonian_spsa_exp)

random_hamiltonian_adam_spsa_exp = {
    'name': 'Random Hamiltonian Adam SPSA Random Experiments',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': 'spsa'},
        'est_shots': {'value': 1},
        'stddev': {'value': 0.2},
        'alpha': {'value': 0.602},
        'gamma': {'value': 0.101},
        'learning_rate': {'value': 1},
        'beta': {'value': 0.99},
        'lmd': {'value': 0.42},
        'zeta': {'value': 0.99},
     }
}
sweep_configs["random_hamiltonian_adam_spsa_exp"] = merge_dict(random_hamiltonian_exp_baseline, random_hamiltonian_adam_spsa_exp)

random_hamiltonian_spsa2_exp = {
    'name': 'Randomized Hamiltonian 2-SPSA Random Experiments',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': '2spsa'},
        'est_shots': {'value': 1},
        'learning_rate': {'value': 0.1},
        'stddev': {'value': 0.2},
        'metric_reg': {'value': 0.001},
     }
}
sweep_configs["random_hamiltonian_spsa2_exp"] = merge_dict(random_hamiltonian_exp_baseline, random_hamiltonian_spsa2_exp)

random_hamiltonian_xnes_exp = {
    'name': 'Randomized Hamiltonian xNES Random Experiments',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': 'xnes'},
        'est_shots': {'value': 2},
        'nu_sigma': {'value': 0.001},
        'nu_b': {'value': 0.001},
        'nu_mu': {'value': 0.1},
        'stddev': {'value': 0.1},
    }
}
sweep_configs["random_hamiltonian_xnes_exp"] = merge_dict(random_hamiltonian_exp_baseline, random_hamiltonian_xnes_exp)

random_hamiltonian_snes_exp = {
    'name': 'Randomized Hamiltonian sNES Random Experiments',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': 'snes'},
        'est_shots': {'value': 2},
        'nu_sigma': {'value': 0.01},
        'nu_mu': {'value': 0.1},
        'stddev': {'value': 0.1},
    }
}
sweep_configs["random_hamiltonian_snes_exp"] = merge_dict(random_hamiltonian_exp_baseline, random_hamiltonian_snes_exp)

random_hamiltonian_ges_exp = {
    'name': 'Randomized Hamiltonian GES Random Experiments',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': 'ges'},
        'est_shots': {'value': 1},
        'learning_rate': {'value': 0.1},
        'explore_tradeoff': {'value': 0.5},
        'grad_scale': {'value': 1},
        'stddev': {'value': 0.1},
        'grad_memory': {'value': 25},
    }
}
sweep_configs["random_hamiltonian_ges_exp"] = merge_dict(random_hamiltonian_exp_baseline, random_hamiltonian_ges_exp)

# ------------------ Hyperparam Sweeps ------------------


# BASELINE - Combine with an experiment-specific dict

hs_baseline = {
    'method': 'random',
    'metric': {'goal': 'minimize', 'name': 'loss'},
    'parameters': 
    {
        'rand_seed': {'values': [7, 13, 42]},
        'print_interval': {'value': 50},
        'device': {'value': 'lightning.qubit'},
     }
}

# OPTIMIZER CONFIGS

spsa_hs = {
    'name': 'SPSA Hyperparam Sweep',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': 'spsa'},
        'est_shots': {'value': 1},
        'stddev': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 1
        },
        'alpha': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 1
        },
        'gamma': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 1
        },
     }
}

adam_spsa_hs = {
    'name': 'Adam SPSA Hyperparam Sweep',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': 'adamspsa'},
        'est_shots': {'value': 1},
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.01,
            'max': 1
        },
        'stddev': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 1
        },
        'alpha': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 1
        },
        'gamma': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 1
        },
        'beta': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 1
        },
        'lmd': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 1
        },
        'zeta': {
            'distribution': 'uniform',
            'min': 0.01,
            'max': 1
        },
     }
}

qnspsa_hs = {
    'name': 'QNSPSA Hyperparam Sweep',
    'parameters': 
    {
        'device': {'value': 'default.qubit'},
        'interface': {'value': 'numpy'},
        'optimizer': {'value': 'pl_qnspsa'},
        'est_shots': {'value': 1},
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1
        },
        'stddev': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1
        },
        'metric_reg': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.1
        },
     }
}

spsa2_hs = {
    'name': '2-SPSA Hyperparam Sweep',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': '2spsa'},
        'est_shots': {'value': 1},
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1
        },
        'stddev': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1
        },
        'metric_reg': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.1
        },
     }
}

xnes_hs = {
    'name': 'xNES Hyperparam Sweep',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': 'xnes'},
        'est_shots': {'value': 2},
        'nu_sigma': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1
        },
        'nu_b': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1
        },
        'nu_mu': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1
        },
        'stddev': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1
        },
     }
}

snes_hs = {
    'name': 'sNES Hyperparam Sweep',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': 'snes'},
        'est_shots': {'value': 2},
        'nu_sigma': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1
        },
        'nu_mu': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1
        },
        'stddev': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1
        },
     }
}

ges_hs = {
    'name': 'GES Hyperparam Sweep',
    'parameters': 
    {
        'interface': {'value': 'torch'},
        'optimizer': {'value': 'ges'},
        'est_shots': {'value': 1},
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1
        },
        'explore_tradeoff': {
            'distribution': 'uniform',
            'min': 0,
            'max': 1
        },
        'grad_scale': {
            'distribution': 'log_uniform_values',
            'min': 0.1,
            'max': 10
        },
        'stddev': {
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 1
        },
        'grad_memory': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 50
        },
     }
}

# SPECIFIC EXPERIMENTS CONFIGS

small_ising_hs = {
    'parameters': 
    {
        'num_qubits': {'value': 3},
        'problem': {'value': 'transverse_ising'},
        'model': {'value': 'rand_layers'},
        'num_layers': {'value': 3},
        'num_params': {'value': 10},
        'ratio_imprim': {'value': 0.3},
        'steps': {'value': 500},
     }
}
small_ising_hs_baseline = merge_dict(hs_baseline, small_ising_hs)
sweep_configs["small_ising_spsa_hs"] = merge_dict(small_ising_hs_baseline, spsa_hs)
sweep_configs["small_ising_spsa_hs"]["name"] = 'Small Ising SPSA Hyperparam Sweep'
sweep_configs["small_ising_adam_spsa_hs"] = merge_dict(small_ising_hs_baseline, adam_spsa_hs)
sweep_configs["small_ising_adam_spsa_hs"]["name"] = 'Small Ising Adam SPSA Hyperparam Sweep'
sweep_configs["small_ising_qnspsa_hs"] = merge_dict(small_ising_hs_baseline, qnspsa_hs)
sweep_configs["small_ising_qnspsa_hs"]["name"] = 'Small Ising QNSPSA Hyperparam Sweep'
sweep_configs["small_ising_spsa2_hs"] = merge_dict(small_ising_hs_baseline, spsa2_hs)
sweep_configs["small_ising_spsa2_hs"]["name"] = 'Small Ising 2-SPSA Hyperparam Sweep'
sweep_configs["small_ising_xnes_hs"] = merge_dict(small_ising_hs_baseline, xnes_hs)
sweep_configs["small_ising_xnes_hs"]["name"] = 'Small Ising xNES Hyperparam Sweep'
sweep_configs["small_ising_snes_hs"] = merge_dict(small_ising_hs_baseline, snes_hs)
sweep_configs["small_ising_snes_hs"]["name"] = 'Small Ising sNES Hyperparam Sweep'
sweep_configs["small_ising_ges_hs"] = merge_dict(small_ising_hs_baseline, ges_hs)
sweep_configs["small_ising_ges_hs"]["name"] = 'Small Ising GES Hyperparam Sweep'

random_hamiltonian_hs = {
    'parameters': 
    {
        'num_qubits': {'value': 10},
        'problem': {'value': 'randomized_hamiltonian'},
        'num_random_singles' : {'value': 10},
        'num_random_doubles' : {'value': 20},
        'model': {'value': 'rand_layers'},
        'num_layers': {'value': 3},
        'num_params': {'value': 10},
        'ratio_imprim': {'value': 0.3},
        'steps': {'value': 500},
     }
}
random_hamiltonian_hs_baseline = merge_dict(hs_baseline, random_hamiltonian_hs)
sweep_configs["random_hamiltonian_spsa_hs"] = merge_dict(random_hamiltonian_hs_baseline, spsa_hs)
sweep_configs["random_hamiltonian_spsa_hs"]["name"] = 'Random Hamiltonian SPSA Hyperparam Sweep'
sweep_configs["random_hamiltonian_adam_spsa_hs"] = merge_dict(random_hamiltonian_hs_baseline, adam_spsa_hs)
sweep_configs["random_hamiltonian_adam_spsa_hs"]["name"] = 'Random Hamiltonian Adam SPSA Hyperparam Sweep'
sweep_configs["random_hamiltonian_qnspsa_hs"] = merge_dict(random_hamiltonian_hs_baseline, qnspsa_hs)
sweep_configs["random_hamiltonian_qnspsa_hs"]["name"] = 'Random Hamiltonian QNSPSA Hyperparam Sweep'
sweep_configs["random_hamiltonian_spsa2_hs"] = merge_dict(random_hamiltonian_hs_baseline, spsa2_hs)
sweep_configs["random_hamiltonian_spsa2_hs"]["name"] = 'Random Hamiltonian 2-SPSA Hyperparam Sweep'
sweep_configs["random_hamiltonian_xnes_hs"] = merge_dict(random_hamiltonian_hs_baseline, xnes_hs)
sweep_configs["random_hamiltonian_xnes_hs"]["name"] = 'Random Hamiltonian xNES Hyperparam Sweep'
sweep_configs["random_hamiltonian_snes_hs"] = merge_dict(random_hamiltonian_hs_baseline, snes_hs)
sweep_configs["random_hamiltonian_snes_hs"]["name"] = 'Random Hamiltonian sNES Hyperparam Sweep'
sweep_configs["random_hamiltonian_ges_hs"] = merge_dict(random_hamiltonian_hs_baseline, ges_hs)
sweep_configs["random_hamiltonian_ges_hs"]["name"] = 'Random Hamiltonian GES Hyperparam Sweep'