# basic nflow train args
import argparse

Y_DIM = 6
Z_DIM = 4   

def add_args(parser):

    # model architecture options
    parser.add_argument('--use_context', type=eval, default=True, choices=[True, False])
    parser.add_argument('--y_dim', default=Y_DIM,
                        help='Dimension of the conditioning variable')
    parser.add_argument('--z_dim', type=int, default=Z_DIM,
                        help='Dimension of the target variable')


    # flow options
    parser.add_argument('--num_splines', type=int, default=4)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--dropout_probability', type=float, default=0.0)
    parser.add_argument('--batch_norm', type=eval, default=True, choices=[True, False])
    parser.add_argument('--num_bins', type=int, default=8)
    parser.add_argument('--tail_bound', type=float, default=3.0)
    parser.add_argument('--tails', type=str, default='linear', choices=['linear', 'logistic', 'exp'])
    parser.add_argument('--num_hidden_channels', type=int, default=128)
    parser.add_argument('--transform_type', type=str, default='rq-autoregressive', choices=['rq-autoregressive', 'rq-coupling'])
    parser.add_argument('--reverse_mask', type=eval, default=False, choices=[True, False])
    parser.add_argument('--permute_mask', type=eval, default=False, choices=[True, False])
    parser.add_argument('--init_identity', type=eval, default=True, choices=[True, False])


    # training options
    parser.add_argument('--n_load_cores', type=int, default=20)
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
    parser.add_argument('--train_limit', type=int, default=500000)
    parser.add_argument('--test_start', type=int, default=500000)
    parser.add_argument('--test_limit', type=int, default=600000)
    

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