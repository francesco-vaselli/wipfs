import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch import nn
from torchinfo import summary

from basic_nflow import create_NDE_model


class Encoder(nn.Module):
    def __init__(self, zdim, input_dim=3, use_deterministic_encoder=False):
        super(Encoder, self).__init__()
        self.use_deterministic_encoder = use_deterministic_encoder
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        if self.use_deterministic_encoder:
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc_bn1 = nn.BatchNorm1d(256)
            self.fc_bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, zdim)
        else:
            # Mapping to [c], cmean
            self.fc1_m = nn.Linear(512, 256)
            self.fc2_m = nn.Linear(256, 128)
            self.fc3_m = nn.Linear(128, zdim)
            self.fc_bn1_m = nn.BatchNorm1d(256)
            self.fc_bn2_m = nn.BatchNorm1d(128)

            # Mapping to [c], cmean
            self.fc1_v = nn.Linear(512, 256)
            self.fc2_v = nn.Linear(256, 128)
            self.fc3_v = nn.Linear(128, zdim)
            self.fc_bn1_v = nn.BatchNorm1d(256)
            self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        if self.use_deterministic_encoder:
            ms = F.relu(self.fc_bn1(self.fc1(x)))
            ms = F.relu(self.fc_bn2(self.fc2(ms)))
            ms = self.fc3(ms)
            m, v = ms, 0
        else:
            m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
            m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
            m = self.fc3_m(m)
            v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
            v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
            v = self.fc3_v(v)

        return m, v


class SimplerEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3, use_deterministic_encoder=False):
        super(SimplerEncoder, self).__init__()
        self.use_deterministic_encoder = use_deterministic_encoder
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        if self.use_deterministic_encoder:
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc_bn1 = nn.BatchNorm1d(256)
            self.fc_bn2 = nn.BatchNorm1d(128)
            self.fc3 = nn.Linear(128, zdim)
        else:
            # Mapping to [c], cmean
            self.fc1_m = nn.Linear(256, 128)
            self.fc2_m = nn.Linear(128, 64)
            self.fc3_m = nn.Linear(64, zdim)
            self.fc_bn1_m = nn.BatchNorm1d(128)
            self.fc_bn2_m = nn.BatchNorm1d(64)

            # Mapping to [c], cmean
            self.fc1_v = nn.Linear(256, 128)
            self.fc2_v = nn.Linear(128, 64)
            self.fc3_v = nn.Linear(64, zdim)
            self.fc_bn1_v = nn.BatchNorm1d(128)
            self.fc_bn2_v = nn.BatchNorm1d(64)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = (self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 128)

        if self.use_deterministic_encoder:
            ms = F.relu(self.fc_bn1(self.fc1(x)))
            ms = F.relu(self.fc_bn2(self.fc2(ms)))
            ms = self.fc3(ms)
            m, v = ms, 0
        else:
            m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
            m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
            m = self.fc3_m(m)
            v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
            v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
            v = self.fc3_v(v)

        return m, v


# Model
class FakeDoubleFlow(nn.Module):
    def __init__(self, args):
        super(FakeDoubleFlow, self).__init__()
        self.input_dim = args["input_dim"]
        self.zdim = args["zdim"]
        self.use_latent_flow = args["use_latent_flow"]
        self.use_deterministic_encoder = args["use_deterministic_encoder"]
        self.prior_weight = args["prior_weight"]
        self.recon_weight = args["recon_weight"]
        self.entropy_weight = args["entropy_weight"]
        self.distributed = args["distributed"]
        self.truncate_std = None
        self.latent_flow_param_dict = args["latent_flow_param_dict"]
        self.reco_flow_param_dict = args["reco_flow_param_dict"]
        self.encoder = SimplerEncoder(
            zdim=args['zdim'],
            input_dim=args['input_dim'],
            use_deterministic_encoder=args['use_deterministic_encoder'],
        )
        self.latent_NDE_model = create_NDE_model(**self.latent_flow_param_dict)
        self.reco_NDE_model = create_NDE_model(**self.reco_flow_param_dict)

        # params printout
        encorder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        latent_NDE_params = sum(p.numel() for p in self.latent_NDE_model.parameters() if p.requires_grad)
        reco_NDE_params = sum(p.numel() for p in self.reco_NDE_model.parameters() if p.requires_grad)
        dummy_bs = 2048
        print("Encoder params: ", summary(self.encoder, input_size=(dummy_bs, self.input_dim))
        print("Latent NDE params: ", summary(self.latent_NDE_model, input_size=(dummy_bs, self.zdim))
        print("Reco NDE params: ", summary(self.reco_NDE_params, input_size=(dummy_bs, 30))
        print("Total params: ", encorder_params + latent_NDE_params + reco_NDE_params)

    """ should not be used with our nflow
    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.cuda(gpu)
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y
    """

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps

    @staticmethod
    def gaussian_entropy(logvar):
        const = 0.5 * float(logvar.size(1)) * (1.0 + np.log(np.pi * 2))
        ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
        return ent

    def multi_gpu_wrapper(self, f):
        self.encoder = f(self.encoder)
        self.latent_NDE_model = f(self.latent_NDE_model)
        self.reco_NDE_model = f(self.reco_NDE_model)

    def make_optimizer(self, args):
        def _get_opt_(params):
            if args["optimizer"] == "adam":
                optimizer = optim.Adam(
                    params,
                    lr=args["lr"],
                    betas=(args["beta1"], args["beta2"]),
                    weight_decay=args["weight_decay"],
                )
            elif args["optimizer"] == "sgd":
                optimizer = torch.optim.SGD(
                    params, lr=args["lr"], momentum=args["momentum"]
                )
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer

        opt = _get_opt_(
            list(self.encoder.parameters())
            + list(self.latent_NDE_model.parameters())
            + list(list(self.reco_NDE_model.parameters()))
        )
        return opt

    # we pass y as conditioning variable
    def forward(self, x, y, N, opt, step, writer=None):
        opt.zero_grad()
        batch_size = x.size(0)
        num_points = x.size(1)
        z_mu, z_sigma = self.encoder(x)
        if self.use_deterministic_encoder:
            z = z_mu + 0 * z_sigma
        else:
            z = self.reparameterize_gaussian(z_mu, z_sigma)

        # Compute H[Q(z|X)]
        if self.use_deterministic_encoder:
            entropy = torch.zeros(batch_size).to(z)
        else:
            entropy = self.gaussian_entropy(z_sigma)

        # Compute the prior probability P(z)
        if self.use_latent_flow:
            """
            w, delta_log_pw = self.latent_cnf(z, None, torch.zeros(batch_size, 1).to(z))
            log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(1, keepdim=True)
            delta_log_pw = delta_log_pw.view(batch_size, 1)
            log_pz = log_pw - delta_log_pw
            """
            log_pz = self.latent_NDE_model.log_prob(z, context=y)
        else:
            log_pz = torch.zeros(batch_size, 1).to(z)

        # Compute the reconstruction likelihood P(X|z)
        z_new = z.view(*z.size())
        z_new = z_new + (log_pz * 0.0).mean()
        # add N of true fakes
        z_new = torch.cat([z_new, N], dim=0)
        """
        y, delta_log_py = self.point_cnf(x, z_new, torch.zeros(batch_size, num_points, 1).to(x))
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        log_px = log_py - delta_log_py
        """
        log_px = self.reco_NDE_model.log_prob(x, context=z_new)

        # Loss
        entropy_loss = -entropy.mean() * self.entropy_weight
        recon_loss = -log_px.mean() * self.recon_weight
        prior_loss = -log_pz.mean() * self.prior_weight
        loss = entropy_loss + prior_loss + recon_loss
        loss.backward()
        opt.step()

        return loss
        """
        # LOGGING (after the training)
        if self.distributed:
            entropy_log = reduce_tensor(entropy.mean())
            recon = reduce_tensor(-log_px.mean())
            prior = reduce_tensor(-log_pz.mean())
        else:
            entropy_log = entropy.mean()
            recon = -log_px.mean()
            prior = -log_pz.mean()

        recon_nats = recon / float(x.size(1) * x.size(2))
        prior_nats = prior / float(self.zdim)

        if writer is not None:
            writer.add_scalar('train/entropy', entropy_log, step)
            writer.add_scalar('train/prior', prior, step)
            writer.add_scalar('train/prior(nats)', prior_nats, step)
            writer.add_scalar('train/recon', recon, step)
            writer.add_scalar('train/recon(nats)', recon_nats, step)

        return {
            'entropy': entropy_log.cpu().detach().item()
            if not isinstance(entropy_log, float) else entropy_log,
            'prior_nats': prior_nats,
            'recon_nats': recon_nats,
        }
        """

    def encode(self, x):
        z_mu, z_sigma = self.encoder(x)
        if self.use_deterministic_encoder:
            return z_mu
        else:
            return self.reparameterize_gaussian(z_mu, z_sigma)

    def decode(self, z, num_points, truncate_std=None):
        # transform points from the prior to a point cloud, conditioned on a shape code
        x = self.reco_NDE_model.sample(num_points, context=z)
        return x

    def sample(
        self,
        y,
        batch_size,
        num_points,
        truncate_std=None,
        truncate_std_latent=None,
        gpu=None,
    ):
        assert (
            self.use_latent_flow
        ), "Sampling requires `self.use_latent_flow` to be True."
        # Generate the shape code from the prior
        z = self.latent_NDE_model.sample(1, context=y)
        # Sample points conditioned on the shape code
        x = self.reco_NDE_model.sample(num_points, context=z)
        return z, x

    def reconstruct(self, x, num_points=None, truncate_std=None):
        num_points = x.size(1) if num_points is None else num_points
        z = self.encode(x)
        _, x = self.decode(z, num_points, truncate_std)
        return x



