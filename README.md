# yeastdnnexplorer

[![gh-pages](https://github.com/BrentLab/yeastdnnexplorer/actions/workflows/docs.yml/badge.svg)](https://github.com/BrentLab/yeastdnnexplorer/actions/workflows/docs.yml)
[![style](https://img.shields.io/badge/%20style-sphinx-0a507a.svg)](https://www.sphinx-doc.org/en/master/usage/index.html)
[![Pytest](https://github.com/BrentLab/yeastdnnexplorer/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BrentLab/yeastdnnexplorer/actions/workflows/ci.yml)
[![gh-pages](https://github.com/BrentLab/yeastdnnexplorer/actions/workflows/docs.yml/badge.svg)](https://github.com/BrentLab/yeastdnnexplorer/actions/workflows/docs.yml)

## Documentation

See [here]() for more complete documentation

## Installation

(no user installation instructions yet)

### Development

1. git clone the repo
1. `cd` into the local version of the repo
1. choose one (or more) of the following

#### vscode

I strongly recommend using vscode in the `devcontainer` environment. This will
ensure that you have all the necessary dependencies installed and configured,
as well as your vscode environment set up for automatic linting/formatting.

By default, the `devcontainer` expects that you have Nvidia GPUs with the
Nvidia Container Tookit installed and docker enabled. If you have Nvidia GPUs,
but not the tookit, see the section on installing with docker:
[Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html?highlight=docker#configuring-docker)

If you don't want GPU access in the devcontainer, you can also change the
value of `dockerComposeFile` from `gpu.yml` to `cpu.yml` in the
[devcontainer.json](.devcontainer/devcontainer.json).

#### poetry

You can also install the dependencies using poetry. I prefer setting the following:

```bash
poetry config virtualenvs.in-project true
```

So that the virtual environments are installed in the project directory as `.venv`

After cloning and `cd`ing into the repo, you can install the dependencies with:

```bash
poetry install
```

#### docker compose

You can create an environment using docker compose. There are two compose
configuration files:

- `gpu.yml`: for use with Nvidia GPUs
- `cpu.yml`: for use without GPUs

You can build the environment with:

```bash
docker-compose -f <gpu/cpu>.yml build
```

After that you can start a shell with:

```bash
docker-compose -f <gpu/cpu>.yml run --rm app bash
```

#### github codespaces

you don't need to clone the repo to your local for this, but you will
need to fork the repo into your local github account. Make sure you change
the `dockerComposeFile` in the [devcontainer.json](.devcontainer/devcontainer.json)
to `cpu.yml` and then launch a codespace.
