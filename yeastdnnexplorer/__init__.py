import os

from dotenv import load_dotenv


def load_environment_variables():
    """Load environment variables from the .env file in the package base directory."""
    # Get the absolute path of the package base directory
    package_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct the path to the .env file in the package base directory
    env_path = os.path.join(package_base_dir, ".env")

    # Load the .env file if it exists
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)


# Load environment variables
load_environment_variables()
