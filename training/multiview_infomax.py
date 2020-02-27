from training.base import RepresentationTrainer
from utils.modules import MIEstimator


######################
# MV InfoMax Trainer #
######################

class MVInfoMaxTrainer(RepresentationTrainer):
    def __init__(self, miest_lr, **params):
        super(MVInfoMaxTrainer, self).__init__(**params)

        # Initialization of the mutual information estimation network
        self.mi_estimator = MIEstimator(self.z_dim, self.z_dim)

        # Adding the parameters of the estimator to the optimizer
        self.opt.add_param_group(
            {'params': self.mi_estimator.parameters(), 'lr': miest_lr}
        )

        # Intialization of the encoder(s)
        # In this example encoder_v1 and encoder_v2 completely share their parameters
        self.encoder_v1 = self.encoder
        self.encoder_v2 = self.encoder_v1

    def _get_items_to_store(self):
        items_to_store = super(MVInfoMaxTrainer, self)._get_items_to_store()

        # Add the mutual information estimator parameters to items_to_store
        items_to_store['mi_estimator'] = self.mi_estimator.state_dict()
        return items_to_store

    def _compute_loss(self, data):
        # Read the two views v1 and v2 and ignore the label y
        v1, v2, _ = data

        # Encode a batch of data
        p_z1_given_v1 = self.encoder_v1(v1)
        p_z2_given_v2 = self.encoder_v2(v2)

        # Sample from the posteriors with reparametrization
        z1 = p_z1_given_v1.rsample()
        z2 = p_z2_given_v2.rsample()

        # Mutual information estimation
        mi_gradient, mi_estimation = self.mi_estimator(z1, z2)
        mi_gradient = mi_gradient.mean()
        mi_estimation = mi_estimation.mean()

        # Logging Mutual Information Estimation
        self._add_loss_item('loss/I_z1_z2', mi_estimation.item())

        # Computing the loss function
        loss = - mi_gradient

        return loss