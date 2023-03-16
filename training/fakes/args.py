import argparse

NONLINEARITIES = ["tanh", "relu", "softplus", "elu", "swish", "square", "identity"]
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
LAYERS = ["ignore", "concat", "concat_v2", "squash", "concatsquash", "scale", "concatscale"]

X_DIM = 30
Y_DIM = 7
Z_DIM = 3   

def add_args(parser):

    # model architecture options
    parser.add_argument('--use_context', type=eval, default=True, choices=[True, False])
    parser.add_argument('--x_dim', type=int, default=X_DIM)
    parser.add_argument('--y_dim', type=int, default=Y_DIM)
    parser.add_argument('--input_dim', type=int, default=30,
                        help='Number of input dimensions (30 as we flatten all fakes)')
    parser.add_argument('--dims', type=str, default='256')
    parser.add_argument('--latent_dims', type=str, default='256')
    parser.add_argument("--num_blocks", type=int, default=1,
                        help='Number of stacked CNFs.')
    parser.add_argument("--latent_num_blocks", type=int, default=1,
                        help='Number of stacked CNFs.')
    parser.add_argument("--layer_type", type=str, default="concatsquash", choices=LAYERS)
    parser.add_argument('--time_length', type=float, default=0.5)
    parser.add_argument('--train_T', type=eval, default=True, choices=[True, False])
    parser.add_argument("--nonlinearity", type=str, default="tanh", choices=NONLINEARITIES)
    parser.add_argument('--use_adjoint', type=eval, default=True, choices=[True, False])
    parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--sync_bn', type=eval, default=False, choices=[True, False])
    parser.add_argument('--bn_lag', type=float, default=0)

    # flow options
    parser.add_argument('--num_steps_maf', type=int, default=6)
    parser.add_argument('--num_steps_arqs', type=int, default=20)
    parser.add_argument('--num_transform_blocks_maf', type=int, default=2)
    parser.add_argument('--num_transform_blocks_arqs', type=int, default=6)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--dropout_probability_maf', type=float, default=0.0)
    parser.add_argument('--dropout_probability_arqs', type=float, default=0.1)
    parser.add_argument('--use_residual_blocks_maf', type=eval, default=False, choices=[True, False])
    parser.add_argument('--use_residual_blocks_arqs', type=eval, default=True, choices=[True, False])
    parser.add_argument('--batch_norm_maf', type=eval, default=True, choices=[True, False])
    parser.add_argument('--batch_norm_arqs', type=eval, default=True, choices=[True, False])
    parser.add_argument('--num_bins', type=int, default=32)
    parser.add_argument('--tail_bound', type=float, default=3.0)
    parser.add_argument('--hidden_dim_maf', type=int, default=128)
    parser.add_argument('--hidden_dim_arqs', type=int, default=198)
    parser.add_argument('--base_transform_type', type=str, default='rq-autoregressive', choices=['rq-autoregressive', 'rq-coupling'])
    parser.add_argument('--transform_type', type=str, default='random-permutation', choices=['random-permutation', 'block-permutation', 'no-permutation'])
    parser.add_argument('--block_size', type=int, default=2)
    parser.add_argument('--mask_type', type=str, default='alternating-binary', choices=['alternating-binary', 'block-binary', 'identity'])
    parser.add_argument('--init_identity', type=eval, default=True, choices=[True, False])


    # training options
    parser.add_argument('--n_load_cores', type=int, default=0)
    parser.add_argument('--zdim', type=int, default=Z_DIM,
                        help='Dimension of the shape code')
    parser.add_argument('--z_dim', type=int, default=Z_DIM,
                        help='Dimension of the shape code')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use', choices=['adam', 'adamax', 'sgd'])
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size (of datasets) for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Latent learning rate for the Adam optimizer.')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Beta1 for Adam.')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for the optimizer.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs for training (default: 500)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for initializing training. ')
    parser.add_argument('--scheduler', type=str, default='linear',
                        help='Type of learning rate schedule')
    parser.add_argument('--exp_decay', type=float, default=1.,
                        help='Learning rate schedule exponential decay rate')
    parser.add_argument('--exp_decay_freq', type=int, default=1,
                        help='Learning rate exponential decay frequency')

    # data options
    parser.add_argument('--no_N', type=eval, default=False, choices=[True, False])
    parser.add_argument('--with_zeros', default=False , action='store_true')
    parser.add_argument('--sorted_dataset', default=False, action='store_true')
    parser.add_argument('--no_rint', type=eval, default=True, choices=[True, False])
    parser.add_argument('--rescale_data', default=False, action='store_true')
    parser.add_argument('--shuffle_train', type=eval, default=True, choices=[True, False])
    parser.add_argument('--train_start', type=int, default=0)
    parser.add_argument('--train_limit', type=int, default=1000000)
    parser.add_argument('--test_start', type=int, default=0)
    parser.add_argument('--test_limit', type=int, default=100000)
    

    # logging and saving frequency
    parser.add_argument('--log_name', type=str, default='saves_fakes', help="Name for the log dir")
    parser.add_argument('--val_freq', type=int, default=5)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)

    # validation options
    parser.add_argument('--validate_at_0', default=False, action='store_true')
    parser.add_argument('--no_validation', default=False, action='store_true',
                        help='Whether to disable validation altogether.')
    parser.add_argument('--save_val_results', default=True, action='store_true',
                        help='Whether to save the validation results.')
    
    # resuming
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to the checkpoint to be loaded.')
    parser.add_argument('--resume' , default=False, action='store_true')
    parser.add_argument('--resume_optimizer', action='store_true',
                        help='Whether to resume the optimizer when resumed training.')
    parser.add_argument('--resume_non_strict', action='store_true',
                        help='Whether to resume in none-strict mode.')
    parser.add_argument('--resume_dataset_mean', type=str, default=None,
                        help='Path to the file storing the dataset mean.')
    parser.add_argument('--resume_dataset_std', type=str, default=None,
                        help='Path to the file storing the dataset std.')

    # device
    parser.add_argument('--device', default='cuda', type=str)
    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use. None means using all available GPUs.')

    return parser


def get_parser():
    # command line args
    parser = argparse.ArgumentParser(description='Flow-based Point Cloud Generation Experiment')
    parser = add_args(parser)
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args