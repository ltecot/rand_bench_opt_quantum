
spsa_hyperparam_sweep = {
    'method': 'random',
    'name': 'SPSA Hyperparam Sweep',
    'metric': {'goal': 'minimize', 'name': 'loss'},
    'parameters': 
    {
        # Basics
        'rand_seed': {'value': 42},
        'rand_seed_model': {'value': 42},
        'rand_seed_problem': {'value': 42},
        'print_interval': {'value': 50},
        'num_qubits': {'value': 3},
        'interface': {'value': 'torch'},
        'problem': {'value': 'transverse_ising'},
        'model': {'value': 'rand_layers'},
        'num_layers': {'value': 3},
        'num_params': {'value': 10},
        'ratio_imprim': {'value': 0.3},
        # Optimizer Params
        'optimizer': {'value': 'spsa'},
        'steps': {'value': 500},
        'est_shots': {'value': 1},
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.001,
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
     }
}