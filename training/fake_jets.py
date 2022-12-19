
# the args dictionary defining all the parameters for the FakeDoubleFlow model
args = {
    'zdim': 128,
    'input_dim': 3,
    'optimizer': 'adam'
    'lr': 0.001,
    'weight_decay': 0.0,
    'beta1': 0.9,
    'beta2': 0.999,
    'entropy_weight': 1.0,
    'prior_weight': 1.0,
    'recon_weight': 1.0,
    'use_deterministic_encoder': False,
    'use_latent_flow': True,
    'latent_flow_param_dict': {
        "input_dim" : 17,
        "context_dim" : 14,
        "num_flow_steps" : 15,

        "base_param_dict" : {
        "num_transform_blocks": 10,
        "activation": "relu",
        "batch_norm": True,
        "num_bins": 128,
        "hidden_dim": 298,
        "block_size": 3,
	    "mask_type" : "identity"
        },

        "transform_type" : "no-permutation" 
    },
    'reco_flow_param_dict': {
                "input_dim" : 17,
        "context_dim" : 14,
        "num_flow_steps" : 15,

        "base_param_dict" : {
        "num_transform_blocks": 10,
        "activation": "relu",
        "batch_norm": True,
        "num_bins": 128,
        "hidden_dim": 298,
        "block_size": 3,
	    "mask_type" : "identity"
        },

        "transform_type" : "no-permutation" 
    },
}