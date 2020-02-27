from utils.modules import MIEstimator
from training.base import RepresentationTrainer


######################
# MV InfoMax Trainer #
######################
class InfoMaxTrainer(RepresentationTrainer):
    def __init__(self, miest_lr=1e-4, **params):
        super(InfoMaxTrainer, self).__init__(**params)

        # Initialization of the mutual information estimation network
        self.mi_estimator = MIEstimator(self.z_dim, 28*28)

        # Adding the parameters of the estimator to the optimizer
        self.opt.add_param_group(
            {'params': self.mi_estimator.parameters(), 'lr': miest_lr}
        )

    def _get_items_to_store(self):
        items_to_store = super(InfoMaxTrainer, self)._get_items_to_store()

        # Add the mutual information estimator parameters to items_to_store
        items_to_store['mi_estimator'] = self.mi_estimator.state_dict()
        return items_to_store

    def _compute_loss(self, data):
        # Ignore the second view and the label
        x, _, _ = data

        # Encode a batch of data
        p_z_given_x = self.encoder(x)

        # Sample from the posteriors with reparametrization
        z = p_z_given_x.rsample()

        # Mutual information estimation
        mi_gradient, mi_estimation = self.mi_estimator(z, x.view(x.shape[0], -1))
        mi_gradient = mi_gradient.mean()
        mi_estimation = mi_estimation.mean()

        # Logging Mutual Information Estimation
        self._add_loss_item('loss/I_z_x', mi_estimation.item())

        # Computing the loss function
        loss = - mi_gradient

        return loss