
# ------------------ Util ------------------

def merge_dict(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, {})
            merge_dict(value, node)
        else:
            destination[key] = value
    return destination

# ------------------ Experiment Sweeps ------------------

exp_baseline = {
    'method': 'grid',
    'metric': {'goal': 'minimize', 'name': 'loss'},
    'parameters': 
    {
        'rand_seed': {'value': None},
        'rand_seed_model': {'values': [5632, 6145, 5123, 4626, 4117, 1558, 9241, 5155, 6193, 5685, 
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
        # 'rand_seed_problem': {'value': None},  # For now this does nothing
        'print_interval': {'value': 50},
        'num_qubits': {'value': 3},
        'problem': {'value': 'transverse_ising'},
        'model': {'value': 'rand_layers'},
        'num_layers': {'value': 3},
        'num_params': {'value': 10},
        'ratio_imprim': {'value': 0.3},
        'steps': {'value': 500},
     }
}

spsa_exp = {
    'name': 'SPSA Random Experiments',
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
merge_dict(exp_baseline, spsa_exp)

# ------------------ Hyperparam Sweeps ------------------

hs_baseline = {
    'method': 'random',
    'metric': {'goal': 'minimize', 'name': 'loss'},
    'parameters': 
    {
        'rand_seed': {'value': 42},
        'rand_seed_model': {'value': 42},
        'rand_seed_problem': {'value': 42},
        'print_interval': {'value': 50},
        'num_qubits': {'value': 3},
        'problem': {'value': 'transverse_ising'},
        'model': {'value': 'rand_layers'},
        'num_layers': {'value': 3},
        'num_params': {'value': 10},
        'ratio_imprim': {'value': 0.3},
        'steps': {'value': 500},
     }
}

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
merge_dict(hs_baseline, spsa_hs)

qnspsa_hs = {
    'name': 'QNSPSA Hyperparam Sweep',
    'parameters': 
    {
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
merge_dict(hs_baseline, qnspsa_hs)