# A Singular Experiment Launcher

Singular is a quality of life package that enables rapid deployment of code on a slurm cluster with singularity installed and password enabled login. Running experiments on your cluster is as simple as a single command in the terminal using singular. See below for an example and install instructions.

## Installation

Singular can be installed using the pip package.

```bash
pip install singular-launcher
```

## Usage

You may configure singular to remember the ssh credentials to your cluster using the following example.

```bash
singular set --ssh-username username --ssh-password password --ssh-host compute.example.com
```

Running your first command on your cluster is then as simple as one line in the terminal.

```bash
singular remote echo "my first command"
```

Certain workloads require uploading certain data files into the singularity image on the remote machine before running experiments. This can be done with the following command.

```bash
singular upload --recursive --exclude "*.pkl" ./local_dir remote_dir/in/image
```

Additionally, you can download files from the remote machine with a single command. The following command will download the results folder inside the remote singularity image to a location in the current local working directory. The remote path is always taken with respect to the singularity image path.

```bash
singular download --recursive --exclude "*.pkl" results ./
```
