import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch import nn

from modded_basic_nflow import create_NDE_model


# Latent Model
class LatentFlow(nn.Module):
    def __init__(self, args):
        super(LatentFlow, self).__init__()
        self.input_dim = args.input_dim
        self.zdim = args.zdim
        self.use_latent_flow = args.use_latent_flow

        self.latent_flow_param_dict = args.latent_flow_param_dict

        self.latent_NDE_model = create_NDE_model(**self.latent_flow_param_dict)
        self.y_dim = args.y_dim
        self.use_context = args.use_context

        # params printout
        latent_NDE_params = sum(
            p.numel() for p in self.latent_NDE_model.parameters() if p.requires_grad
        )

        # print("Encoder params: ", self.encoder)
        # print("Latent NDE params: ", self.latent_NDE_model)
        # print("Reco NDE params: ", self.reco_NDE_model)
        print("Total params: ", latent_NDE_params)

    def make_optimizer(self, args):
        def _get_opt_(params):
            if args.optimizer == "adam":
                optimizer = optim.Adam(
                    params,
                    lr=args.lr_latent,
                    betas=(args.beta1, args.beta2),
                    weight_decay=args.weight_decay,
                )
            elif args.optimizer == "sgd":
                optimizer = torch.optim.SGD(
                    params, lr=args.lr_latent, momentum=args.momentum
                )
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer

        opt = _get_opt_(list(self.latent_NDE_model.parameters()))
        return opt

    # we pass y as conditioning variable
    def forward(self, y, z, opt, step, epoch, writer=None, val=False):
        batch_size = y.size(0)
        y_size = self.y_dim

        if val==False:
            opt.zero_grad()

        # Compute the prior probability P(z)

            if self.use_latent_flow:
                """
                w, delta_log_pw = self.latent_cnf(z, None, torch.zeros(batch_size, 1).to(z))
                log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(1, keepdim=True)
                delta_log_pw = delta_log_pw.view(batch_size, 1)
                log_pz = log_pw - delta_log_pw
                """
                # print(z.size(), y.size())
                if self.use_context:
                    log_pz = self.latent_NDE_model.log_prob(z, context=y.view(-1, y_size))
                else:
                    log_pz = self.latent_NDE_model.log_prob(z)
            else:
                log_pz = torch.zeros(batch_size, 1).to(z)

            prior_loss = -log_pz.mean()
            loss = prior_loss

            loss.backward()
            opt.step()

            # LOGGING (after the training)
            prior = -log_pz.mean()

            prior_nats = prior / float(self.zdim)

        elif val==True:
            with torch.no_grad():
                if self.use_latent_flow:
                    # print(z.size(), y.size())
                    if self.use_context:
                        log_pz = self.latent_NDE_model.log_prob(z, context=y.view(-1, y_size))
                    else:
                        log_pz = self.latent_NDE_model.log_prob(z)
                else:
                    log_pz = torch.zeros(batch_size, 1).to(z)

                prior_loss = -log_pz.mean()
                loss = prior_loss

                # LOGGING (after the training)
                prior = -log_pz.mean()

                prior_nats = prior / float(self.zdim)

        if writer is not None and val is False:
            writer.add_scalar("train/prior", prior, step)
            writer.add_scalar("train/prior(nats)", prior_nats, step)

        if writer is not None and val is True:
            writer.add_scalar("val/prior", prior, step)
            writer.add_scalar("val/prior(nats)", prior_nats, step)

        return prior_nats

    def sample(self, num_samples, context=None):
        if self.use_context:
            z = self.latent_NDE_model.sample(num_samples, context=context)
        else:
            z = self.latent_NDE_model.sample(num_samples)
        return z


# Reco model
class RecoFlow(nn.Module):
    def __init__(self, args):
        super(RecoFlow, self).__init__()
        self.input_dim = args.input_dim
        self.zdim = args.zdim

        self.reco_flow_param_dict = args.reco_flow_param_dict

        self.reco_NDE_model = create_NDE_model(**self.reco_flow_param_dict)

        # params printout
        reco_NDE_params = sum(
            p.numel() for p in self.reco_NDE_model.parameters() if p.requires_grad
        )

        # print("Encoder params: ", self.encoder)
        # print("Latent NDE params: ", self.latent_NDE_model)
        # print("Reco NDE params: ", self.reco_NDE_model)
        print("Total params: ", reco_NDE_params)

    def make_optimizer(self, args):
        def _get_opt_(params):
            if args.optimizer == "adam":
                optimizer = optim.Adam(
                    params,
                    lr=args.lr_reco,
                    betas=(args.beta1, args.beta2),
                    weight_decay=args.weight_decay,
                )
            elif args.optimizer == "sgd":
                optimizer = torch.optim.SGD(
                    params, lr=args.lr_reco, momentum=args.momentum
                )
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer

        opt = _get_opt_(list(self.reco_NDE_model.parameters()))
        return opt

    def forward(self, x, z, opt, step, epoch, writer=None, val=False):

        if val==False:
            opt.zero_grad()

            # Compute the P(x|z)
            log_px = self.reco_NDE_model.log_prob(x, context=z.view(-1, self.zdim))
            posterior_loss = -log_px.mean()
            loss = posterior_loss

            loss.backward()
            opt.step()

            # LOGGING (after the training)
            posterior = -log_px.mean()

            posterior_nats = posterior / float(self.input_dim)

        elif val==True:
            with torch.no_grad():
                # Compute the P(x|z)
                log_px = self.reco_NDE_model.log_prob(x, context=z.view(-1, self.zdim))
                posterior_loss = -log_px.mean()
                loss = posterior_loss

                # LOGGING (after the training)
                posterior = -log_px.mean()

                posterior_nats = posterior / float(self.input_dim)

        if writer is not None and val is False:
            writer.add_scalar("train/posterior", posterior, step)
            writer.add_scalar("train/posterior(nats)", posterior_nats, step)
        if writer is not None and val is True:
            writer.add_scalar("val/posterior", posterior, step)
            writer.add_scalar("val/posterior(nats)", posterior_nats, step)

        return posterior_nats

    def sample(self, num_samples, z):
        x = self.reco_NDE_model.sample(num_samples, context=z)
        return x
