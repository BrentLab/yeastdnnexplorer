"""Script to inspect the parameters of a trained model (passed in via a checkpoint
file)"""

import argparse

from yeastdnnexplorer.ml_models.simple_model import SimpleModel


def inspect_model_experiment(checkpoint_file_path: str) -> None:
    """
    Runs the simple experiement to inspect the parameters of a trained model.

    :param checkpoint_file_path: The path to the model checkpoint file that we want to
        inspect
    :type checkpoint_file_path: str

    """

    # load the model from the checkpoint
    model = SimpleModel.load_from_checkpoint(checkpoint_path=checkpoint_file_path)

    print("Model Hyperparameters===========================================")
    print(model.hparams)

    print("Model Parameters================================================")
    for name, param in model.named_parameters():
        print(f"{name}: {param.size()}")
        print(f"\t{param.data}")


def parse_args_for_inspect_model_experiment() -> argparse.Namespace:
    """
    Parses the command line arguments for the inspect_model_experiment function Fails
    with error message if the required argument (checkpoint_file) is not provided.

    :return: The parsed command line arguments
    :rtype: argparse.Namespace

    """
    parser = argparse.ArgumentParser(description="Inspcting Model Parameters")
    parser.add_argument(
        "--checkpoint_file", type=str, action="store", required=True
    )  # this will be the checkpoint file that we want to inspect
    args = parser.parse_args()
    return args


def main() -> None:
    """Main method to run he experiment for inspecting the parameters of a trained
    model."""
    args = parse_args_for_inspect_model_experiment()

    # use default values if flag not present in command line arguments
    checkpoint_file_path = args.checkpoint_file

    inspect_model_experiment(checkpoint_file_path)


if __name__ == "__main__":
    main()
