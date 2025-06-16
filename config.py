class Configuration:
    initialize_variance = 1e-1
    opt_einsum_optimize = 'greedy'
    contraction_engine = 'opt_einsum' # or 'QCTN'