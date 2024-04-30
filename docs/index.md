# yeastdnnexplorer

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![style](https://img.shields.io/badge/%20style-sphinx-0a507a.svg)](https://www.sphinx-doc.org/en/master/usage/index.html)
[![Pytest](https://github.com/BrentLab/yeastdnnexplorer/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BrentLab/yeastdnnexplorer/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BrentLab/yeastdnnexplorer/graph/badge.svg?token=D2AB7IUY7F)](https://codecov.io/gh/BrentLab/yeastdnnexplorer)
[![gh-pages](https://github.com/BrentLab/yeastdnnexplorer/actions/workflows/docs.yml/badge.svg)](https://github.com/BrentLab/yeastdnnexplorer/actions/workflows/docs.yml)

## Introduction

`yeastdnnexplorer` is intended to serve as a development environment for exploring
different DNN models to infer the relationship between transcription factors and
target genes using binding and perturbation expression data.

## Installation

This repo has not yet been added to PyPI. See the developer installation below.

### Development

1. git clone the repo
1. `cd` into the local version of the repo
1. choose one (or more) of the following (only poetry currently supported)

<!-- #### vscode

I strongly recommend using vscode in the `devcontainer` environment. This will
ensure that you have all the necessary dependencies installed and configured,
as well as your vscode environment set up for automatic linting/formatting.

By default, the `devcontainer` expects that you have Nvidia GPUs with the
Nvidia Container Tookit installed and docker enabled. If you have Nvidia GPUs,
but not the tookit, see the section on installing with docker:
[Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html?highlight=docker#configuring-docker)

If you don't want GPU access in the devcontainer, you can also change the
value of `dockerComposeFile` from `gpu.yml` to `cpu.yml` in the
[devcontainer.json](.devcontainer/devcontainer.json). -->

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

<!-- #### docker compose

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
to `cpu.yml` and then launch a codespace. -->

#### mkdocs

The documentation is build with mkdocs:

##### Commands

After building the environment with poetry, you can use `poetry run` or a poetry shell
to execute the following:

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

##### Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.