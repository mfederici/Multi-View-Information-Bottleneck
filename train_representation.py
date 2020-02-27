import os
import yaml
import argparse
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, RandomAffine, ToTensor
from torch.utils.data import DataLoader

from utils.data import PixelCorruption, AugmentedDataset
from utils.evaluation import evaluate, split
import training as training_module


parser = argparse.ArgumentParser()
parser.add_argument("experiment_dir", type=str,
                    help="Full path to the experiment directory. Logs and checkpoints will be stored in this location")
parser.add_argument("--config-file", type=str, default=None, help="Path to the .yaml training configuration file.")
parser.add_argument("--data-dir", type=str, default='.', help="Root path for the datasets.")
parser.add_argument("--no-logging", action="store_true", help="Disable tensorboard logging")
parser.add_argument("--overwrite", action="store_true",
                    help="Force the over-writing of the previous experiment in the specified directory.")
parser.add_argument("--device", type=str, default="cuda",
                    help="Device on which the experiment is executed (as for tensor.device). Specify 'cpu' to "
                         "force execution on CPU.")
parser.add_argument("--num-workers", type=int, default=8,
                    help="Number of CPU threads used during the data loading procedure.")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size used for the experiments.")
parser.add_argument("--load-model-file", type=str, default=None,
                    help="Checkpoint to load for the experiments. Note that the specified configuration file needs "
                         "to be compatible with the checkpoint.")
parser.add_argument("--checkpoint-every", type=int, default=50, help="Frequency of model checkpointing (in epochs).")
parser.add_argument("--backup-every", type=int, default=5, help="Frequency of model backups (in epochs).")
parser.add_argument("--evaluate-every", type=int, default=5, help="Frequency of model evaluation.")
parser.add_argument("--epochs", type=int, default=1000, help="Total number of training epochs")

args = parser.parse_args()

logging = not args.no_logging
experiment_dir = args.experiment_dir
data_dir = args.data_dir
config_file = args.config_file
overwrite = args.overwrite
device = args.device
num_workers = args.num_workers
batch_size = args.batch_size
load_model_file = args.load_model_file
checkpoint_every = args.checkpoint_every
backup_every = args.backup_every
evaluate_every = args.evaluate_every
epochs = args.epochs

# Check if the experiment directory already contains a model
pretrained = os.path.isfile(os.path.join(experiment_dir, 'model.pt')) \
             and os.path.isfile(os.path.join(experiment_dir, 'config.yml'))


if pretrained and not (config_file is None) and not overwrite:
    raise Exception("The experiment directory %s already contains a trained model, please specify a different "
                    "experiment directory or remove the --config-file option to resume training or use the --overwrite"
                    "flag to force overwriting")

resume_training = pretrained and not overwrite


if resume_training:
    load_model_file = os.path.join(experiment_dir, 'model.pt')
    config_file = os.path.join(experiment_dir, 'config.yml')

if logging:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=experiment_dir)
else:
    os.makedirs(experiment_dir, exist_ok=True)
    writer = None

# Load the configuration file
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)

# Copy it to the experiment folder
with open(os.path.join(experiment_dir, 'config.yml'), 'w') as file:
    yaml.dump(config, file)

# Instantiating the trainer according to the specified configuration
TrainerClass = getattr(training_module, config['trainer'])
trainer = TrainerClass(writer=writer, **config['params'])

# Resume the training if specified
if load_model_file:
    trainer.load(load_model_file)

# Moving the models to the specified device
trainer.to(device)

###########
# Dataset #
###########
# Loading the MNIST dataset
mnist_dir = os.path.join(data_dir, 'MNIST')
train_set = MNIST(mnist_dir, download=True, train=True, transform=ToTensor())
test_set = MNIST(mnist_dir, download=True, train=False, transform=ToTensor())

# Defining the augmentations
t = Compose([
    RandomAffine(degrees=15,
                 translate=[0.1, 0.1],
                 scale=[0.9, 1.1],
                 shear=15),  # Small affine transformations
    ToTensor(),              # Conversion to torch tensor
    PixelCorruption(0.8)     # PixelCorruption with keep probability 80%
])

# Creating the multi-view dataset using the augmentation class defined by t
mv_train_set = AugmentedDataset(MNIST(mnist_dir, download=True), t)

# Initialization of the data loader
train_loader = DataLoader(mv_train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Select a subset 100 samples (10 for each per label)
train_subset = split(train_set, 100, 'Balanced')

##########

checkpoint_count = 1

for epoch in tqdm(range(epochs)):
    for data in tqdm(train_loader):
        trainer.train_step(data)

    if epoch % evaluate_every == 0:
        # Compute train and test_accuracy of a logistic regression
        train_accuracy, test_accuracy = evaluate(encoder=trainer.encoder, train_on=train_subset, test_on=test_set,
                                                 device=device)
        if not (writer is None):
            writer.add_scalar(tag='evaluation/train_accuracy', scalar_value=train_accuracy, global_step=trainer.iterations)
            writer.add_scalar(tag='evaluation/test_accuracy', scalar_value=test_accuracy, global_step=trainer.iterations)

        tqdm.write('Train Accuracy: %f' % train_accuracy)
        tqdm.write('Test Accuracy: %f' % test_accuracy)

    if epoch % checkpoint_every == 0:
        tqdm.write('Storing model checkpoint')
        while os.path.isfile(os.path.join(experiment_dir, 'checkpoint_%d.pt' % checkpoint_count)):
            checkpoint_count += 1

        trainer.save(os.path.join(experiment_dir, 'checkpoint_%d.pt' % checkpoint_count))
        checkpoint_count += 1

    if epoch % backup_every == 0:
        tqdm.write('Updating the model backup')
        trainer.save(os.path.join(experiment_dir, 'model.pt'))
