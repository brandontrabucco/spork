# A Singular Experiment Launcher

Singular is a quality of life package that enables rapid deployment of code on a slurm cluster with singularity installed and password enabled login. Running experiments on your cluster is as simple as a single command in the terminal using singular. See below for an example and install instructions.

## Installation

Singular can be installed using pip.

```bash
pip install singular-launcher
```

## Usage

You may configure singular to remember the ssh credentials to your remote cluster using the following example in the terminal.

```bash
singular set --ssh-username username --ssh-password password --ssh-host compute.example.com
```

Running your first command on your cluster is then as simple as one line in the terminal.

```bash
singular remote echo "my first command"
```