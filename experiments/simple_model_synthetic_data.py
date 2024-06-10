"""Script to train our simple model on synthetic data and save the best model based on
validation loss."""

import argparse
from argparse import Namespace

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from yeastdnnexplorer.data_loaders.synthetic_data_loader import SyntheticDataLoader
from yeastdnnexplorer.ml_models.simple_model import SimpleModel

# Callback to save the best model based on validation loss
best_model_checkpoint = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    filename="best-model-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
)

# Callback to save checkpoints every 5 epochs, regardless of performance
periodic_checkpoint = ModelCheckpoint(
    filename="periodic-{epoch:02d}",
    every_n_epochs=2,
    save_top_k=-1,  # Setting -1 saves all checkpoints
)
# Need to configure the loggers we're going to use
tb_logger = TensorBoardLogger("logs/tensorboard_logs")
csv_logger = CSVLogger("logs/csv_logs")


def simple_model_synthetic_data_experiment(
    batch_size: int,
    lr: float,
    max_epochs: int,
    using_random_seed: bool,
    accelerator: str,
) -> None:
    """
    Trains a SimpleModel on synthetic data and saves the best model based on validation
    loss. Defines an instance of Trainer, which is used to train the model with the
    given dataModule. While much of the training process is captured via logging, we
    also print the test results at the end of training. We don't need to do assrtions
    for type checking, as this was done in the parse_args_for_synthetic_data_experiment
    function.

    :param batch_size: The batch size to use for training
    :type batch_size: int
    :param lr: The learning rate to use for training
    :type lr: float
    :param max_epochs: The maximum number of epochs to train for
    :type max_epochs: int
    :param using_random_seed: Whether or not to use a random seed for reproducibility
    :type using_random_seed: bool
    :param accelerator: The accelerator to use for training (e.g. 'gpu', 'cpu')
    :type accelerator: str

    """

    data_module = SyntheticDataLoader(
        batch_size=batch_size,
        num_genes=1000,
        bound=[0.1, 0.15, 0.2, 0.25, 0.3],
        n_sample=[1, 1, 2, 2, 4],
        val_size=0.1,
        test_size=0.1,
        random_state=42,
    )

    num_tfs = sum(data_module.n_sample)  # sum of all n_sample is the number of TFs

    model = SimpleModel(input_dim=num_tfs, output_dim=num_tfs, lr=lr)
    trainer = Trainer(
        max_epochs=max_epochs,
        deterministic=using_random_seed,
        accelerator=accelerator,
        callbacks=[best_model_checkpoint, periodic_checkpoint],
        logger=[tb_logger, csv_logger],
    )
    trainer.fit(model, data_module)

    test_results = trainer.test(model, datamodule=data_module)
    print(
        test_results
    )  # this prints all metrics that were logged during the test phase


def parse_args_for_synthetic_data_experiment() -> Namespace:
    """
    Parses command line arguments for the synthetic data experiment.

    :return: The command line arguments
    :rtype: Namespace
    :raises ValueError: If batch_size is not an integer greater than 0
    :raises ValueError: If lr is not a float greater than 0
    :raises ValueError: If max_epochs is not an integer greater than 0
    :raises ValueError: If random_seed is not an integer greater than or equal to 0
    :raises ValueError: If gpus is not an integer greater than or equal to 0

    """
    parser = argparse.ArgumentParser(
        description="Simple Model Synthetic Data Experiment"
    )
    parser.add_argument("--batch_size", action="store", type=int)
    parser.add_argument("--lr", action="store", type=float)
    parser.add_argument("--max_epochs", action="store", type=int)
    parser.add_argument("--random_seed", action="store", type=int)
    parser.add_argument("--gpus", action="store", type=int)

    # note that this performs the type checking needed
    # so we don't need assertion checks for that
    args = parser.parse_args()

    # assert correct values
    if args.batch_size and args.batch_size < 1:
        raise ValueError("batch_size must be an integer greater than 0")
    if args.lr and args.lr <= 0:
        raise ValueError("lr must be a float greater than 0")
    if args.max_epochs and args.max_epochs < 1:
        raise ValueError("max_epochs must be an integer greater than 0")
    if args.random_seed and args.random_seed < 0:
        raise ValueError("random_seed must be an integer greater than or equal to 0")
    if args.gpus and args.gpus < 0:
        raise ValueError("gpus must be an integer greater than or equal to 0")

    return args


def main() -> None:
    """
    Main method to run the experiment for training the simple model using the syntheetic
    data loader.

    Saves the best model based on validation loss.

    """
    args = parse_args_for_synthetic_data_experiment()

    # use default values if flag not present in command line arguments
    batch_size = args.batch_size or 32
    lr = args.lr or 0.01
    max_epochs = args.max_epochs or 10
    random_seed = args.random_seed or 42
    gpus = args.gpus or 0

    # set random seed for reproducibility
    seed_everything(random_seed)

    # run the experiment
    simple_model_synthetic_data_experiment(
        batch_size=batch_size,
        lr=lr,
        max_epochs=max_epochs,
        using_random_seed=True,
        accelerator="gpu" if (gpus > 0) else "cpu",
    )


if __name__ == "__main__":
    main()
