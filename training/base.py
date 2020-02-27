import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer

from utils.modules import Encoder

import torch.optim as optimizer_module


# utility function to initialize an optimizer from its name
def init_optimizer(optimizer_name, params):
    assert hasattr(optimizer_module, optimizer_name)
    OptimizerClass = getattr(optimizer_module, optimizer_name)
    return OptimizerClass(params)


##########################
# Generic training class #
##########################
class Trainer(nn.Module):
    def __init__(self, log_loss_every=10, writer=None):
        super(Trainer, self).__init__()
        self.iterations = 0

        self.writer = writer
        self.log_loss_every = log_loss_every

        self.loss_items = {}

    def get_device(self):
        return list(self.parameters())[0].device

    def train_step(self, data):
        # Set all the models in training mode
        self.train(True)

        # Log the values in loss_items every log_loss_every iterations
        if not (self.writer is None):
            if (self.iterations + 1) % self.log_loss_every == 0:
                self._log_loss()

        # Move the data to the appropriate device
        device = self.get_device()

        for i, item in enumerate(data):
            data[i] = item.to(device)

        # Perform the training step and update the iteration count
        self._train_step(data)
        self.iterations += 1

    def _add_loss_item(self, name, value):
        assert isinstance(name, str)
        assert isinstance(value, float) or isinstance(value, int)

        if not (name in self.loss_items):
            self.loss_items[name] = []

        self.loss_items[name].append(value)

    def _log_loss(self):
        # Log the expected value of the items in loss_items
        for key, values in self.loss_items.items():
            self.writer.add_scalar(tag=key, scalar_value=np.mean(values), global_step=self.iterations)
            self.loss_items[key] = []

    def save(self, model_path):
        items_to_save = self._get_items_to_store()
        items_to_save['iterations'] = self.iterations

        # Save the model and increment the checkpoint count
        torch.save(items_to_save, model_path)

    def load(self, model_path):
        items_to_load = torch.load(model_path)
        for key, value in items_to_load.items():
            assert hasattr(self, key)
            attribute = getattr(self, key)

            # Load the state dictionary for the stored modules and optimizers
            if isinstance(attribute, nn.Module) or isinstance(attribute, Optimizer):
                attribute.load_state_dict(value)

                # Move the optimizer parameters to the same correct device.
                # see https://github.com/pytorch/pytorch/issues/2830 for further details
                if isinstance(attribute, Optimizer):
                    device = list(value['state'].values())[0]['exp_avg'].device # Hack to identify the device
                    for state in attribute.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)

            # Otherwise just copy the value
            else:
                setattr(self, key, value)

    def _get_items_to_store(self):
        return dict()

    def _train_step(self, data):
        raise NotImplemented()


##########################
# Representation Trainer #
##########################

# Generic class to train an model with a (stochastic) neural network encoder

class RepresentationTrainer(Trainer):
    def __init__(self, z_dim, optimizer_name='Adam', encoder_lr=1e-4, **params):
        super(RepresentationTrainer, self).__init__(**params)

        self.z_dim = z_dim

        # Intialization of the encoder
        self.encoder = Encoder(z_dim)

        self.opt = init_optimizer(optimizer_name, [
            {'params': self.encoder.parameters(), 'lr': encoder_lr},
        ])

    def _get_items_to_store(self):
        items_to_store = super(RepresentationTrainer, self)._get_items_to_store()

        # store the encoder and optimizer parameters
        items_to_store['encoder'] = self.encoder.state_dict()
        items_to_store['opt'] = self.opt.state_dict()

        return items_to_store

    def _train_step(self, data):
        loss = self._compute_loss(data)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def _compute_loss(self, data):
        raise NotImplemented