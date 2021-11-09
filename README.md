# Spork: Helping You Run Slurm Jobs

Spork is a quality of life package that enables rapid deployment of code on a slurm cluster with singularity installed and password enabled login. Running experiments on your cluster is as simple as a single command in the terminal using spork. See below for an example and install instructions.

## Installation

Spork can be installed using the pip package.

```bash
pip install spork-cli
```

## Usage

You may configure spork to remember the ssh credentials to your cluster using the following example.

```bash
spork set --ssh-username username --ssh-password password --ssh-host compute.example.com
```

Running your first command on your cluster is then as simple as one line in the terminal.

```bash
spork remote echo "my first command"
```

Certain workloads require uploading certain data files into the singularity image on the remote machine before running experiments. This can be done with the following command.

```bash
spork upload --recursive --exclude "*.pkl" ./local_dir remote_dir/in/image
```

Additionally, you can download files from the remote machine with a single command. The following command will download the results folder inside the remote singularity image to a location in the current local working directory. The remote path is always taken with respect to the singularity image path.

```bash
spork download --recursive --exclude "*.pkl" results ./
```

## Experimentation

A typical experiment creation pipeline involved working on code locally, testing the code locally, then running in on a server. This can be done easily using spork.

```bash
python do_my_experiment.py
```

Once your code is ready for deployment, tell spork how to install your code from github.

```bash
spork set --git-url https://github.com/username/repo
spork set --git-target /code/repo
spork set --install-command "pip install -e {git_target}"
```

Then, point spork to where your code working directory is stored locally.

```bash
spork set --sync-from /home/username/repo
```

Then test your code in a local copy of the singularity to make sure it works as expected.

```bash
spork local --sync python /code/repo/do_my_experiment.py
```

Then run it on the cluster.

```bash
spork remote --sync python /code/repo/do_my_experiment.py
```

In this example, the sync flag tells spork to copy code from your repository on the local disk to code folder in your remote singularity image. This is especially helpful when changes aren't committed.

