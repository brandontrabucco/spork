import time
import sys
import os
import subprocess
import paramiko
import pickle as pkl
import click
import json

from itertools import product
from pexpect import spawn, EOF
from typing import List, Sequence

RECIPE_TEMPLATE = r"""Bootstrap: {bootstrap}
From: {bootstrap_from}

%post

echo 'export PATH="/nvbin:$PATH"' >> /environment
echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> /environment
echo 'export CPATH="/usr/local/cuda/include:$CPATH"' >> /environment
echo 'export LD_LIBRARY_PATH="/nvlib:$LD_LIBRARY_PATH"' >> /environment
echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> /environment
echo 'export CUDA_HOME="/usr/local/cuda"' >> /environment

touch /bin/nvidia-smi
touch /usr/bin/nvidia-smi
touch /usr/bin/nvidia-debugdump
touch /usr/bin/nvidia-persistenced
touch /usr/bin/nvidia-cuda-mps-control
touch /usr/bin/nvidia-cuda-mps-server

mkdir -p /etc/dcv
mkdir -p /var/lib/dcv-gl
mkdir -p /usr/lib64
mkdir -p /code
mkdir -p /results

{before_apt_commands}

apt-get update -y && {apt_packages}

{post_apt_commands}

wget https://repo.anaconda.com/archive/Anaconda3-{anaconda_version}-Linux-x86_64.sh -O /anaconda3.sh
bash /anaconda3.sh -b -p /anaconda3 && rm /anaconda3.sh
. /anaconda3/etc/profile.d/conda.sh
conda update -y conda

{before_env_commands}

conda create -y -n {env_name} python={python_version} {env_create_arguments}
conda activate {env_name}

{post_env_commands}

chmod -R 777 /code
chmod -R 777 /results
chmod -R 777 /anaconda3"""

# arguments for setting up the package environment in singularity
DEFAULT_BOOTSTRAP = "docker"
DEFAULT_BOOTSTRAP_FROM = "nvidia/cuda:11.3.1-cudnn8-devel-ubuntu16.04"
DEFAULT_APT_PACKAGES = ("unzip", "htop", "wget",
                        "git", "vim", "cmake", "gcc", "g++")

# arguments that describe the conda environment to build
DEFAULT_ANACONDA_VERSION = "2021.05"
DEFAULT_PYTHON_VERSION = "3.8"
DEFAULT_ENV_NAME = "venv"
DEFAULT_ENV_CREATE_ARGUMENTS = \
    "pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch"

# a default location for the singularity image and singularity recipe
DEFAULT_RECIPE = "experiment.recipe"
DEFAULT_LOCAL_IMAGE = "experiment.sif"
DEFAULT_REMOTE_IMAGE = "experiment.sif"

# commands that are run before and after apt packages are installed
DEFAULT_BEFORE_APT_COMMANDS = ()
DEFAULT_POST_APT_COMMANDS = ()

# commands that are run before and after the conda environment is created
DEFAULT_BEFORE_ENV_COMMANDS = ()
DEFAULT_POST_ENV_COMMANDS = ()

# information about how to sync code before running an experiment
DEFAULT_SYNC_WITH = ()
DEFAULT_SYNC_TARGET = ()
DEFAULT_EXCLUDE_FROM_SYNC = ()

# commands that are run at various stages of initialization
DEFAULT_SINGULARITY_INIT_COMMANDS = ()
DEFAULT_SLURM_INIT_COMMANDS = ()

# a template for running experiment commands in the container
SINGULARITY_EXEC_TEMPLATE = \
    "singularity exec --nv -w {image} bash -c \"{singularity_command}\""

# a template for launching an experiment using a slurm scheduler
SLURM_SRUN_TEMPLATE = "sbatch --cpus-per-task={num_cpus} \
--gres=gpu:{num_gpus} --mem={memory}g \
--time={num_hours}:00:00 -p {partition} --wrap=\'{slurm_command}\'"

# credentials for logging in to the remote host using ssh
DEFAULT_SSH_USERNAME = "username"
DEFAULT_SSH_PASSWORD = "password"
DEFAULT_SSH_HOST = "matrix.ml.cmu.edu"
DEFAULT_SSH_PORT = 22
DEFAULT_SSH_SLEEP_SECONDS = 0.005

# the amount of seconds to wait by default for a command ran by pexpect
DEFAULT_PROCESS_TIMEOUT = 10800


def add_separator(elements: Sequence[str]):
    for i, element in enumerate(elements):
        yield element + ("" if i == len(elements) - 1 else (
            " " if element.strip().endswith("&") else " && "))


class ExperimentConfig(object):
    """Create an experiment singularity image that manages packages and
    runs experiments in a reproducible and distributable container,
    using a set of package configurations for each experiment.

    Arguments:

    ssh_username: str
        a string representing the username of the account to use when
        connecting to the remote host using ssh.
    ssh_password: str
        a string representing the password of the account to use when
        connecting to the remote host using ssh.
    ssh_host: str
        a string representing the host name of the remote machine on
        which to upload a singularity image and launch jobs.
    ssh_port: int
        an integer representing the port on the remote machine to use
        when connecting to the machine to launch jobs.

    recipe: str
        the location on the disk to write a singularity recipe file
        which will be used later to build a singularity image.
    local_image: str
        the location on the disk to write a singularity image, which will
        be launched when running experiments.
    remote_image: str
        the location on the host to write a singularity image, which will
        be launched when running experiments.

    before_apt_commands: List[str]
        a list of commands to run while building the singularity image
        before apt packages are installed.
    post_apt_commands: List[str]
        a list of commands to run while building the singularity image
        after all apt packages have been installed.
    before_env_commands: List[str]
        a list of commands to run while building the singularity image
        before conda is downloaded and the env is created.
    post_env_commands: List[str]
        a list of commands to run while building the singularity image
        after conda is downloaded and the env has been created.

    bootstrap: str
        whether to bootstrap this singularity image from docker, and
        is set to 'docker' if this is the case.
    bootstrap_from: str
        the source to bootstrap from, which is the name of a docker
        container is bootstrapping from docker as above.
    apt_packages: List[str]
        a list of strings representing the names of apt packages to be
        installed in the current singularity image.

    anaconda_version: str
        the version number of the anaconda package to install, which
        can be set to 2021.05 as a simple default.
    python_version: str
        the version number of the python interpreter to install, which
        can be set to 3.7 as a simple default.
    env_name: str
        the name of the conda environment to build for this experiment
        which can simply be the name of the code-base.
    env_create_packages: str
        a string representing the names and channels of conda packages to
        be installed when creating the associated conda environment.

    sync_with: List[str]
        a string representing the path on disk where uncommitted code is
        stored and can be copied before starting experiments.
    sync_target: List[str]
        a string representing the path on disk where the code will be
        synced into, and experiments will be ran from.
    exclude_from_sync: List[str]
        a string representing the file pattern of files to exclude when
        synchronizing code with the singularity image.

    slurm_init_commands: List[str]
        a list of strings representing commands that are run within the
        slurm node before starting a singularity container.
    singularity_init_commands: List[str]
        a list of strings representing commands that are run within the
        singularity container before starting an experiment.

    """

    def __init__(self, ssh_username: str = DEFAULT_SSH_USERNAME,
                 ssh_password: str = DEFAULT_SSH_PASSWORD,
                 ssh_host: str = DEFAULT_SSH_HOST,
                 ssh_port: int = DEFAULT_SSH_PORT,
                 recipe: str = DEFAULT_RECIPE,
                 local_image: str = DEFAULT_LOCAL_IMAGE,
                 remote_image: str = DEFAULT_REMOTE_IMAGE,
                 before_apt_commands: List[str] = DEFAULT_BEFORE_APT_COMMANDS,
                 post_apt_commands: List[str] = DEFAULT_POST_APT_COMMANDS,
                 before_env_commands: List[str] = DEFAULT_BEFORE_ENV_COMMANDS,
                 post_env_commands: List[str] = DEFAULT_POST_ENV_COMMANDS,
                 bootstrap: str = DEFAULT_BOOTSTRAP,
                 bootstrap_from: str = DEFAULT_BOOTSTRAP_FROM,
                 apt_packages: List[str] = DEFAULT_APT_PACKAGES,
                 anaconda_version: str = DEFAULT_ANACONDA_VERSION,
                 python_version: str = DEFAULT_PYTHON_VERSION,
                 env_name: str = DEFAULT_ENV_NAME,
                 env_create_arguments: str = DEFAULT_ENV_CREATE_ARGUMENTS,
                 sync_with: List[str] = DEFAULT_SYNC_WITH,
                 sync_target: List[str] = DEFAULT_SYNC_TARGET,
                 exclude_from_sync: List[str] = DEFAULT_EXCLUDE_FROM_SYNC,
                 slurm_init_commands: List[str] = DEFAULT_SLURM_INIT_COMMANDS,
                 singularity_init_commands:
                 List[str] = DEFAULT_SINGULARITY_INIT_COMMANDS):
        """Create an experiment singularity image that manages packages and
        runs experiments in a reproducible and distributable container,
        using a set of package configurations for each experiment.

        Arguments:

        ssh_username: str
            a string representing the username of the account to use when
            connecting to the remote host using ssh.
        ssh_password: str
            a string representing the password of the account to use when
            connecting to the remote host using ssh.
        ssh_host: str
            a string representing the host name of the remote machine on
            which to upload a singularity image and launch jobs.
        ssh_port: int
            an integer representing the port on the remote machine to use
            when connecting to the machine to launch jobs.

        recipe: str
            the location on the disk to write a singularity recipe file
            which will be used later to build a singularity image.
        local_image: str
            the location on the disk to write a singularity image, which will
            be launched when running experiments.
        remote_image: str
            the location on the host to write a singularity image, which will
            be launched when running experiments.

        before_apt_commands: List[str]
            a list of commands to run while building the singularity image
            before apt packages are installed.
        post_apt_commands: List[str]
            a list of commands to run while building the singularity image
            after all apt packages have been installed.
        before_env_commands: List[str]
            a list of commands to run while building the singularity image
            before conda is downloaded and the env is created.
        post_env_commands: List[str]
            a list of commands to run while building the singularity image
            after conda is downloaded and the env has been created.

        bootstrap: str
            whether to bootstrap this singularity image from docker, and
            is set to 'docker' if this is the case.
        bootstrap_from: str
            the source to bootstrap from, which is the name of a docker
            container is bootstrapping from docker as above.
        apt_packages: List[str]
            a list of strings representing the names of apt packages to be
            installed in the current singularity image.

        anaconda_version: str
            the version number of the anaconda package to install, which
            can be set to 2021.05 as a simple default.
        python_version: str
            the version number of the python interpreter to install, which
            can be set to 3.7 as a simple default.
        env_name: str
            the name of the conda environment to build for this experiment
            which can simply be the name of the code-base.
        env_create_packages: str
            a string representing the names and channels of conda packages to
            be installed when creating the associated conda environment.

        sync_with: List[str]
            a string representing the path on disk where uncommitted code is
            stored and can be copied before starting experiments.
        sync_target: List[str]
            a string representing the path on disk where the code will be
            synced into, and experiments will be ran from.
        exclude_from_sync: List[str]
            a string representing the file pattern of files to exclude when
            synchronizing code with the singularity image.

        slurm_init_commands: List[str]
            a list of strings representing commands that are run within the
            slurm node before starting a singularity container.
        singularity_init_commands: List[str]
            a list of strings representing commands that are run within the
            singularity container before starting an experiment.

        """

        # arguments for the ssh login credentials of the host
        self.ssh_username = ssh_username
        self.ssh_password = ssh_password
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port

        # locations for a singularity recipe and image to be written
        self.recipe = recipe
        self.local_image = local_image
        self.remote_image = remote_image

        # additional commands to run when building the singularity image
        self.before_apt_commands = before_apt_commands
        self.post_apt_commands = post_apt_commands
        self.before_env_commands = before_env_commands
        self.post_env_commands = post_env_commands

        # global arguments for the singularity image package environment
        self.bootstrap = bootstrap
        self.bootstrap_from = bootstrap_from
        self.apt_packages = apt_packages

        # arguments that specify the package environment for the source code
        self.anaconda_version = anaconda_version
        self.python_version = python_version
        self.env_name = env_name
        self.env_create_arguments = env_create_arguments

        # commands that specify how to update code within the image
        self.sync_with = sync_with
        self.sync_target = sync_target
        self.exclude_from_sync = exclude_from_sync

        # commands that are run at various stages of initialization
        self.slurm_init_commands = slurm_init_commands
        self.singularity_init_commands = singularity_init_commands

    def recipe_exists(self) -> bool:
        """Utility function that checks the local disk for whether a
        singularity recipe with the given name already exists on the disk
        at the desired location, and if so returns true.

        Returns:

        recipe_exists: bool
            a boolean that returns True if the singularity recipe with the
            specified name already exists on the disk.

        """

        return os.path.exists(self.recipe)  # exists at this location

    def local_image_exists(self) -> bool:
        """Utility function that checks the local disk for whether a
        singularity image with the given name already exists on the disk
        at the desired location, and if so returns true.

        Returns:

        image_exists: bool
            a boolean that returns True if the singularity image with the
            specified name already exists on the disk.

        """

        return os.path.exists(self.local_image)  # exists at this location

    @staticmethod
    def remote_path_exists(sftp: paramiko.SFTPClient,
                           remote_path: str) -> bool:
        """Utility function that checks the remote host for whether a
        particular path with the provided name already exists on the host
        at the desired location, and if so returns true.

        Arguments:

        sftp: paramiko.SFTPClient
            an instance of the paramiko SFTPClient that is already open
            and connected to the remote machine.
        remote_path: str
            a string representing the path of the file or directory on the
            remote machine to check and test for existence.

        Returns:

        path_exists: bool
            a boolean that returns True if the given path with the specified
            name already exists on the host machine.

        """

        # check if the file exists on the host by attempting to obtain file
        # information using the stat command
        try:
            time.sleep(DEFAULT_SSH_SLEEP_SECONDS)
            sftp.stat(remote_path)  # check if the remote path exists
        except IOError:
            return False  # stat failed and the file does not exist
        else:
            return True  # stat finished and the file is present

    def remote_image_exists(self, client: paramiko.SSHClient = None) -> bool:
        """Utility function that checks the remote host for whether a
        singularity image with the given name already exists on the host
        at the desired location, and if so returns true.

        Returns:

        image_exists: bool
            a boolean that returns True if the singularity image with the
            specified name already exists on the host.

        """

        if client is None:
            # open an ssh connection to the remote host by logging in using
            # the provided username and password for that machine
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(self.ssh_host, self.ssh_port,
                           username=self.ssh_username,
                           password=self.ssh_password,
                           look_for_keys=False, allow_agent=False)

        # open an sftp client and check if a file exists on the remote host
        with client.open_sftp() as sftp:
            return self.remote_path_exists(sftp, self.remote_image)

    def write_singularity_recipe(self):
        """Using the provided class attributes, write a singularity recipe
        to the disk, which will be used in a later stage to build a
        singularity image for performing experiments.

        """

        # create an installation command for all the provided apt packages
        apt_packages = ("apt-get install -y {apt_packages}"
                        .format(apt_packages=" ".join(self.apt_packages)))

        # write a singularity recipe file to the disk at the desired path
        with open(self.recipe, "w") as recipe_file:
            recipe_file.write(RECIPE_TEMPLATE.format(
                bootstrap=self.bootstrap,
                bootstrap_from=self.bootstrap_from,
                before_apt_commands="\n".join(self.before_apt_commands),
                apt_packages=apt_packages,
                post_apt_commands="\n".join(self.post_apt_commands),
                anaconda_version=self.anaconda_version,
                before_env_commands="\n".join(self.before_env_commands),
                python_version=self.python_version,
                env_name=self.env_name,
                env_create_arguments=self.env_create_arguments,
                post_env_commands="\n".join(self.post_env_commands)))

    def write_singularity_image(self, **kwargs):
        """Using the provided class attributes, generate a singularity
        recipe file and build a singularity image that will be used to run
        experiments in an isolated package environment.

        """

        # if the recipe does not exist locally then write it first
        if not self.recipe_exists():
            self.write_singularity_recipe()

        # build the singularity image using the singularity api
        from spython.main import Client
        Client.build(recipe=self.recipe, image=self.local_image,
                     sudo=False, sandbox=True,
                     options=["--fakeroot", "--force"], **kwargs)

    @staticmethod
    def local_rsync(source_path: str, destination_path: str,
                    recursive: bool = False, exclude: str = ""):
        """Using the provided class attributes, generate and run a command in
        a bash shell that will call the rsync utility in order to copy files
        from a source location to a destination on the local machine.

        Arguments:

        source_path: str
            a string representing the path on the local disk to a file or
            directory that will be copied from.
        destination_path: str
            a string representing the path on the local disk to a file or
            directory that will be copied into from a source.
        recursive: bool
            a boolean that controls whether rsync will be called with the
            recursive option to copy a directory.
        exclude: str
            a string representing the file pattern for files to exclude
            from the copy, such as data files.

        """

        # handle multiple exclude patterns separated by whitespace
        command = [y for x in exclude.split(" ") for y in ["--exclude", x]]
        command.extend([source_path, destination_path])

        # spawn a child process that copies files to the host using rsync
        child = spawn("rsync", ["-ra" if recursive else "-a", "--progress",
                                *command], encoding='utf-8')

        # print outputs of the process to the terminal but not passwords
        child.logfile_read = sys.stdout

        # catch when the process finishes because it outputs the EOF token
        child.expect(EOF, timeout=DEFAULT_PROCESS_TIMEOUT)

    def remote_rsync(self, source_path: str, destination_path: str,
                     recursive: bool = False, exclude: str = "",
                     source_is_remote: bool = True,
                     destination_is_remote: bool = True):
        """Using the provided class attributes, generate and run a command in
        a bash shell that will call the rsync utility in order to copy files
        from a source location to a destination on the host machine.

        Arguments:

        source_path: str
            a string representing the path on the local disk to a file or
            directory that will be copied from.
        destination_path: str
            a string representing the path on the local disk to a file or
            directory that will be copied into from a source.
        recursive: bool
            a boolean that controls whether rsync will be called with the
            recursive option to copy a directory.
        exclude: str
            a string representing the file pattern for files to exclude
            from the copy, such as data files.

        source_is_remote: bool
            a boolean that controls whether the source file pattern is
            considered to be on the host machine.
        destination_is_remote: bool
            a boolean that controls whether the destination file pattern is
            considered to be on the host machine.

        """

        # ensure that at least one of the paths is on the remote machine
        assert source_is_remote or destination_is_remote, \
            "use local_rsync instead if not connecting to a server"

        # determine if files are downloaded from or uploaded to the host
        if source_is_remote:
            source_path = "{username}@{host}:{path}".format(
                username=self.ssh_username,
                host=self.ssh_host, path=source_path)

        # determine if files are downloaded from or uploaded to the host
        if destination_is_remote:
            destination_path = "{username}@{host}:{path}".format(
                username=self.ssh_username,
                host=self.ssh_host, path=destination_path)

        # handle multiple exclude patterns separated by whitespace
        command = [y for x in exclude.split(" ") for y in ["--exclude", x]]
        command.extend(["-e", "ssh -p {}".format(self.ssh_port),
                        source_path, destination_path])

        # spawn a child process that copies files to the host using rsync
        child = spawn("rsync", ["-ra" if recursive else "-a", "--progress",
                                *command], encoding='utf-8')

        # print outputs of the process to the terminal but not passwords
        child.logfile_read = sys.stdout

        # expect the host to prompt our client for a kuberos password
        child.expect("{username}@{host}'s password:"
                     .format(username=self.ssh_username, host=self.ssh_host))

        # once we have been prompted for a password enter it into the stdin
        time.sleep(DEFAULT_SSH_SLEEP_SECONDS)
        child.sendline(self.ssh_password)

        # catch when the process finishes because it outputs the EOF token
        child.expect(EOF, timeout=DEFAULT_PROCESS_TIMEOUT)

    def upload_singularity_image(self, rebuild: bool = False):
        """Using the provided class attributes, generate and run a command in
        a bash shell that will write a singularity container and copy it
        from the local disk to a remote host machine.

        Arguments:

        rebuild: bool
            a boolean that controls whether the singularity image should be
            rebuilt even if it already exists on the disk.

        """

        # if the recipe does not exist locally then write it first
        if not self.local_image_exists() or rebuild:
            self.write_singularity_image()  # build the singularity image

        # copy the singularity recipe file to the host
        self.remote_rsync(os.path.join(self.local_image, "."),
                          self.remote_image, source_is_remote=False,
                          destination_is_remote=True, recursive=True)

    def run_in_singularity(self, *commands: str,
                           image: str = DEFAULT_REMOTE_IMAGE) -> str:
        """Using the provided class attributes, generate a command that can
        be executed in a bash shell to start a singularity container and
        run experiments using the installed research code in that container.

        Arguments:

        commands: List[str]
            a list of strings representing commands that are run within the
            container once all setup commands are finished.

        Returns:

        run_command: str
            a string representing a command that can be executed in the
            terminal in order to run experiments using singularity.

        """

        singularity_command = "".join(add_separator(
            [". /anaconda3/etc/profile.d/conda.sh",
             "conda activate {}".format(self.env_name)] +
            list(self.singularity_init_commands) + list(commands)))
        return SINGULARITY_EXEC_TEMPLATE.format(
            singularity_command=singularity_command, image=image)

    def local_run(self, *commands: str, rebuild: bool = False):
        """Generate and run a command in the bash shell that starts a
        singularity container locally and runs commands in that container
        and prints outputs to the standard output stream.

        Arguments:

        commands: List[str]
            a list of strings representing commands that are run within the
            container once all setup commands are finished.
        rebuild: bool
            a boolean that controls whether the singularity image should be
            rebuilt even if it already exists on the disk.

        """

        if rebuild:  # if being rebuilt delete the existing one
            stdout = os.popen("rm {}; rm -rf {}"
                              .format(self.recipe, self.local_image))
            for line in iter(stdout.readline, ""):
                print(line, end="")  # prints even if not finished

        # if the recipe does not exist locally then write it first
        if rebuild or not self.local_image_exists():
            self.write_singularity_image()  # build the singularity image

        # copy the local code directory to the local singularity image
        for sync_with, sync_target, exclude_from_sync in zip(
                self.sync_with, self.sync_target, self.exclude_from_sync):
            self.local_rsync(os.path.join(sync_with, "."), os.path.join(
                self.local_image, sync_target[1:]),
                             recursive=True, exclude=exclude_from_sync)

        # start an experiment locally using a local singularity container
        stdout = os.popen(self.run_in_singularity(*commands,
                                                  image=self.local_image))

        # print the output from the terminal as the command runs
        for line in iter(stdout.readline, ""):
            print(line, end="")  # prints even if not yet finished

    def run_in_slurm(self, *commands: str, partition: str = "russ_reserved",
                     num_cpus: int = 4, num_gpus: int = 1,
                     memory: int = 16, num_hours: int = 8,
                     image: str = DEFAULT_REMOTE_IMAGE) -> str:
        """Using the provided class attributes, generate a command that can
        be executed in a bash shell to start a slurm job that runs
         experiments using a singularity container on a remote machine.

        Arguments:

        commands: List[str]
            a list of strings representing commands that are run within the
            container once all setup commands are finished.
        image: str
            a string representing the path on the remote machine where a
            singularity image can be loaded from.

        num_cpus: int
            an integer representing the number of cpu cores that will be
            allocated by slurm to the generated slurm job.
        num_gpus: int
            an integer representing the number of gpu nodes that will be
            allocated by slurm to the generated slurm job.
        memory: int
            an integer representing the amount of memory that will be
            allocated by slurm to the generated slurm job.
        num_hours: int
            an integer representing the amount of time the slurm job will be
            allowed to run before forcibly terminating.
        partition: str
            a string that represents the slurm partition of machines to use
            when scheduling a slurm job on the host machine.

        Returns:

        run_command: str
            a string representing a command that can be executed in the
            terminal in order to run experiments using singularity.

        """

        slurm_command = "".join(add_separator(
            list(self.slurm_init_commands) +
            [self.run_in_singularity(*commands, image=image)]))
        return SLURM_SRUN_TEMPLATE.format(
            partition=partition, num_cpus=num_cpus, num_gpus=num_gpus,
            num_hours=num_hours, memory=memory, slurm_command=slurm_command)

    def remote_run(self, *commands: str, exclude_nodes: str = None, rebuild: bool = False,
                   sweep_params: List[str] = (), sweep_values: List[str] = (),
                   partition: str = "russ_reserved", num_cpus: int = 4,
                   num_gpus: int = 1, memory: int = 16, num_hours: int = 8):
        """Generate and run a command in the bash shell that starts a
        singularity container remotely and runs commands in that container
        and prints outputs to the standard output stream.

        Arguments:

        commands: List[str]
            a list of strings representing commands that are run within the
            container once all setup commands are finished.
        rebuild: bool
            a boolean that controls whether the singularity image should be
            rebuilt even if it already exists on the disk.

        sweep_params: List[str]
            list of strings representing names of a grid of parameters of
            the specified command that will be searched and replaced.
        sweep_values: List[str]
            list of strings representing values of a grid of parameters of
            the specified command that will be searched and replaced.

        num_cpus: int
            an integer representing the number of cpu cores that will be
            allocated by slurm to the generated slurm job.
        num_gpus: int
            an integer representing the number of gpu nodes that will be
            allocated by slurm to the generated slurm job.
        memory: int
            an integer representing the amount of memory that will be
            allocated by slurm to the generated slurm job.
        num_hours: int
            an integer representing the amount of time the slurm job will be
            allowed to run before forcibly terminating.
        partition: str
            a string that represents the slurm partition of machines to use
            when scheduling a slurm job on the host machine.

        """

        # open an ssh connection to the remote host by logging in using the
        # provided username and password for that machine
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(self.ssh_host, self.ssh_port,
                       username=self.ssh_username, password=self.ssh_password,
                       look_for_keys=False, allow_agent=False)

        if rebuild:  # if the image is being rebuilt delete the existing image
            self.remote_shell("rm -rf {}"
                              .format(self.remote_image), client=client)

        # if the image does not exist on the remote then upload it first
        if rebuild or not self.remote_image_exists(client=client):
            self.upload_singularity_image()  # build the singularity image

        # copy the local code directory to the remote singularity image
        for sync_with, sync_target, exclude_from_sync in zip(
                self.sync_with, self.sync_target, self.exclude_from_sync):
            self.remote_rsync(os.path.join(sync_with, "."), os.path.join(
                self.remote_image, sync_target[1:]),
                              recursive=True, exclude=exclude_from_sync,
                              source_is_remote=False, destination_is_remote=True)

        # generate a template slurm command that will be run on the cluster
        slurm_template = self.run_in_slurm(  # with the target node parameters
            *commands, image=self.remote_image,
            partition=partition, num_cpus=num_cpus,
            num_gpus=num_gpus, num_hours=num_hours, memory=memory)

        # if we are excluding certain nodes on the cluster
        if exclude_nodes is not None:  # then add an exclude flag to slurm
            slurm_template = slurm_template.replace(
                "sbatch", f"sbatch --exclude={exclude_nodes}")

        # iterate through every possible assignment of the parameters on an
        # n-dimensional grid of hyper-parameters
        for assignment in product(*map(lambda v: v.split(","), sweep_values)):

            # fill in the slurm template with hyper-parameter values
            slurm_command = slurm_template  # by replacing names with values
            for param_name, param_value in zip(sweep_params, assignment):
                slurm_command = slurm_template.replace(param_name, param_value)

            # sleep to avoid sending too many commands to the server
            time.sleep(DEFAULT_SSH_SLEEP_SECONDS)

            # generate a command to launch a remote experiment using slurm
            stdout = client.exec_command(slurm_command, get_pty=True)[1]

            # print the output from the terminal as the command runs
            for line in iter(stdout.readline, ""):
                print(line, end="")  # prints even if not yet finished

    def remote_shell(self, *commands: str, client: paramiko.SSHClient = None,
                     watch: bool = False, interval: float = 1.0):
        """Run a set of commands on the remote machine, which can be used to
        check on jobs that are scheduled or running on the host, and also to
        cancel jobs that are scheduled or running.

        Arguments:

        commands: List[str]
            a list of strings representing commands that are run on the host
            machine through an ssh connecting.
        watch: bool
            a boolean that controls whether the remote commands should be
            executed repeatedly at a specified interval.
        interval: float
            a float that represents the amount of time in seconds between
            successive commands when watch is set to True.

        """

        # a function that makes printed output look like the watch utility
        def clear_and_print_header():
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Every {interval}s: {commands}\n"
                  .format(interval=interval, commands=commands))

        if client is None:
            # open an ssh connection to the remote host by logging in using
            # the provided username and password for that machine
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(self.ssh_host, self.ssh_port,
                           username=self.ssh_username,
                           password=self.ssh_password,
                           look_for_keys=False, allow_agent=False)

        # generate a command to that runs on the remote machine over ssh
        time.sleep(DEFAULT_SSH_SLEEP_SECONDS)
        commands = "".join(add_separator(commands))
        stdout = client.exec_command(commands, get_pty=True)[1]

        # print intermediate outputs into the terminal as the command runs
        for i, line in enumerate(iter(stdout.readline, "")):
            if i == 0 and watch:
                clear_and_print_header()
            print(line, end="")  # prints even if not yet finished

        while watch:  # repeat the commands at an interval
            # run the provided commands again in the shell on the host
            time.sleep(interval)
            stdout = client.exec_command(commands, get_pty=True)[1]

            # print intermediate outputs into terminal as the command runs
            for i, line in enumerate(iter(stdout.readline, "")):
                if i == 0:
                    clear_and_print_header()
                print(line, end="")  # prints even if not finished


# the default location for a config file to be stored on the local disk
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.pkl")


class PersistentExperimentConfig(object):
    """Create a persistent wrapper around the ExperimentConfig class that
    enables saving and loading the config multiple times when the
    parameters are changed and experiments are launched.

    """

    def __init__(self, clear: bool = False,
                 storage_path: str = DEFAULT_CONFIG, **kwargs):
        """Create a persistent wrapper around the ExperimentConfig class that
        enables saving and loading the config multiple times when the
        parameters are changed and experiments are launched.

        Arguments:

        storage_path: str
            a string representing the location on the disk where a config
            file will be saved when running jobs on a cluster.

        """

        # get the location of the saved experiment configuration
        self.experiment_config = None
        self.storage_path = storage_path

        # if an experiment configuration has not previously been created
        # then create a new one and save it at the target location
        if clear or not os.path.exists(self.storage_path):
            with open(self.storage_path, "wb") as f:
                pkl.dump(ExperimentConfig(**kwargs), f)  # save a default

    def __enter__(self):
        """When entering the scope of a with statement using the persistent
        experiment configuration object, load existing configs from the
        local disk at the expected disk location.

        Returns:

        config: ExperimentConfig
            an instance of the ExperimentConfig class, that represents the
            settings used when launching experiments on a cluster.

        """

        # load the config from the local disk and return a pointer to it
        with open(self.storage_path, "rb") as f:
            self.experiment_config = pkl.load(f)
            return self.experiment_config

    def __exit__(self, _type, value, traceback):
        """When exiting the scope of a with statement using the persistent
        experiment configuration object, save the object with its new
        changes to the disk at the expected disk location.

        """

        # save the updated config to the disk at the standard location
        with open(self.storage_path, "wb") as f:
            pkl.dump(self.experiment_config, f)
            self.experiment_config = None


@click.group()
def command_line_interface():
    pass  # a default command group that click uses for exposing a cli


@command_line_interface.command()
@click.option('--ssh-username', type=str, default=None)
@click.option('--ssh-password', type=str, default=None)
@click.option('--ssh-host', type=str, default=None)
@click.option('--ssh-port', type=int, default=None)
@click.option('--recipe', type=str, default=None)
@click.option('--local-image', type=str, default=None)
@click.option('--remote-image', type=str, default=None)
@click.option('--before-apt-commands', type=str, default=None, multiple=True)
@click.option('--no-before-apt-commands', is_flag=True)
@click.option('--post-apt-commands', type=str, default=None, multiple=True)
@click.option('--no-post-apt-commands', is_flag=True)
@click.option('--before-env-commands', type=str, default=None, multiple=True)
@click.option('--no-before-env-commands', is_flag=True)
@click.option('--post-env-commands', type=str, default=None, multiple=True)
@click.option('--no-post-env-commands', is_flag=True)
@click.option('--bootstrap', type=str, default=None)
@click.option('--bootstrap-from', type=str, default=None)
@click.option('--apt-packages', type=str, default=None, multiple=True)
@click.option('--no-apt-packages', is_flag=True)
@click.option('--anaconda-version', type=str, default=None)
@click.option('--python-version', type=str, default=None)
@click.option('--env-name', type=str, default=None)
@click.option('--env-create-arguments', type=str, default=None)
@click.option('--sync-with', type=str, default=None, multiple=True)
@click.option('--no-sync-with', is_flag=True)
@click.option('--sync-target', type=str, default=None, multiple=True)
@click.option('--no-sync-target', is_flag=True)
@click.option('--exclude-from-sync', type=str, default=None, multiple=True)
@click.option('--no-exclude-from-sync', is_flag=True)
@click.option('--slurm-init-commands', type=str, default=None, multiple=True)
@click.option('--no-slurm-init-commands', is_flag=True)
@click.option('--singularity-init-commands', type=str, default=None, multiple=True)
@click.option('--no-singularity-init-commands', is_flag=True)
def set(ssh_username: str = DEFAULT_SSH_USERNAME,
        ssh_password: str = DEFAULT_SSH_PASSWORD,
        ssh_host: str = DEFAULT_SSH_HOST,
        ssh_port: int = DEFAULT_SSH_PORT,
        recipe: str = DEFAULT_RECIPE,
        local_image: str = DEFAULT_LOCAL_IMAGE,
        remote_image: str = DEFAULT_REMOTE_IMAGE,
        before_apt_commands: List[str] = DEFAULT_BEFORE_APT_COMMANDS,
        no_before_apt_commands: bool = False,
        post_apt_commands: List[str] = DEFAULT_POST_APT_COMMANDS,
        no_post_apt_commands: bool = False,
        before_env_commands: List[str] = DEFAULT_BEFORE_ENV_COMMANDS,
        no_before_env_commands: bool = False,
        post_env_commands: List[str] = DEFAULT_POST_ENV_COMMANDS,
        no_post_env_commands: bool = False,
        bootstrap: str = DEFAULT_BOOTSTRAP,
        bootstrap_from: str = DEFAULT_BOOTSTRAP_FROM,
        apt_packages: List[str] = DEFAULT_APT_PACKAGES,
        no_apt_packages: bool = False,
        anaconda_version: str = DEFAULT_ANACONDA_VERSION,
        python_version: str = DEFAULT_PYTHON_VERSION,
        env_name: str = DEFAULT_ENV_NAME,
        env_create_arguments: str = DEFAULT_ENV_CREATE_ARGUMENTS,
        sync_with: List[str] = DEFAULT_SYNC_WITH,
        no_sync_with: bool = False,
        sync_target: List[str] = DEFAULT_SYNC_TARGET,
        no_sync_target: bool = False,
        exclude_from_sync: List[str] = DEFAULT_EXCLUDE_FROM_SYNC,
        no_exclude_from_sync: bool = False,
        slurm_init_commands: List[str] = DEFAULT_SLURM_INIT_COMMANDS,
        no_slurm_init_commands: bool = False,
        singularity_init_commands:
        List[str] = DEFAULT_SINGULARITY_INIT_COMMANDS,
        no_singularity_init_commands: bool = False):
    """Assign new parameters to the persistent experiment config by loading
    the config from the disk and accepting new parameters for the config
    via a command line interface exposes to the shell.

    Arguments:

    ssh_username: str
        a string representing the username of the account to use when
        connecting to the remote host using ssh.
    ssh_password: str
        a string representing the password of the account to use when
        connecting to the remote host using ssh.
    ssh_host: str
        a string representing the host name of the remote machine on
        which to upload a singularity image and launch jobs.
    ssh_port: int
        an integer representing the port on the remote machine to use
        when connecting to the machine to launch jobs.

    recipe: str
        the location on the disk to write a singularity recipe file
        which will be used later to build a singularity image.
    local_image: str
        the location on the disk to write a singularity image, which will
        be launched when running experiments.
    remote_image: str
        the location on the host to write a singularity image, which will
        be launched when running experiments.

    before_apt_commands: List[str]
        a list of commands to run while building the singularity image
        before apt packages are installed.
    post_apt_commands: List[str]
        a list of commands to run while building the singularity image
        after all apt packages have been installed.
    before_env_commands: List[str]
        a list of commands to run while building the singularity image
        before conda is downloaded and the env is created.
    post_env_commands: List[str]
        a list of commands to run while building the singularity image
        after conda is downloaded and the env has been created.

    bootstrap: str
        whether to bootstrap this singularity image from docker, and
        is set to 'docker' if this is the case.
    bootstrap_from: str
        the source to bootstrap from, which is the name of a docker
        container is bootstrapping from docker as above.
    apt_packages: List[str]
        a list of strings representing the names of apt packages to be
        installed in the current singularity image.

    anaconda_version: str
        the version number of the anaconda package to install, which
        can be set to 2021.05 as a simple default.
    python_version: str
        the version number of the python interpreter to install, which
        can be set to 3.7 as a simple default.
    env_name: str
        the name of the conda environment to build for this experiment
        which can simply be the name of the code-base.
    env_create_packages: str
        a string representing the names and channels of conda packages to
        be installed when creating the associated conda environment.

    sync_with: List[str]
        a string representing the path on disk where uncommitted code is
        stored and can be copied before starting experiments.
    sync_target: List[str]
        a string representing the path on disk where the code will be
        synced into, and experiments will be ran from.
    exclude_from_sync: List[str]
        a string representing the file pattern of files to exclude when
        synchronizing code with the singularity image.

    slurm_init_commands: List[str]
        a list of strings representing commands that are run within the
        slurm node before starting a singularity container.
    singularity_init_commands: List[str]
        a list of strings representing commands that are run within the
        singularity container before starting an experiment.

    """

    with PersistentExperimentConfig() as config:

        if ssh_username is not None:
            config.ssh_username = ssh_username
        if ssh_password is not None:
            config.ssh_password = ssh_password
        if ssh_host is not None:
            config.ssh_host = ssh_host
        if ssh_port is not None:
            config.ssh_port = ssh_port

        if recipe is not None:
            config.recipe = recipe
        if local_image is not None:
            config.local_image = local_image
        if remote_image is not None:
            config.remote_image = remote_image

        if len(before_apt_commands) > 0:
            config.before_apt_commands = \
                () if no_before_apt_commands else before_apt_commands
        if len(post_apt_commands) > 0:
            config.post_apt_commands = \
                () if no_post_apt_commands else post_apt_commands
        if len(before_env_commands) > 0:
            config.before_env_commands = \
                () if no_before_env_commands else before_env_commands
        if len(post_env_commands) > 0:
            config.post_env_commands = \
                () if no_post_env_commands else post_env_commands

        if bootstrap is not None:
            config.bootstrap = bootstrap
        if bootstrap_from is not None:
            config.bootstrap_from = bootstrap_from
        if len(apt_packages) > 0:
            config.apt_packages = \
                () if no_apt_packages else apt_packages

        if anaconda_version is not None:
            config.anaconda_version = anaconda_version
        if python_version is not None:
            config.python_version = python_version
        if env_name is not None:
            config.env_name = env_name
        if env_create_arguments is not None:
            config.env_create_arguments = env_create_arguments

        if len(sync_with) > 0:
            config.sync_with = \
                () if no_sync_with else sync_with
        if len(sync_target) > 0:
            config.sync_target = \
                () if no_sync_target else sync_target
        if len(exclude_from_sync) > 0:
            config.exclude_from_sync = \
                () if no_exclude_from_sync else exclude_from_sync

        if len(slurm_init_commands) > 0:
            config.slurm_init_commands = \
                () if no_slurm_init_commands else slurm_init_commands
        if len(singularity_init_commands) > 0:
            config.singularity_init_commands = \
                () if no_singularity_init_commands \
                    else singularity_init_commands


@command_line_interface.command()
@click.option('--ssh-username', is_flag=True)
@click.option('--ssh-password', is_flag=True)
@click.option('--ssh-host', is_flag=True)
@click.option('--ssh-port', is_flag=True)
@click.option('--recipe', is_flag=True)
@click.option('--local-image', is_flag=True)
@click.option('--remote-image', is_flag=True)
@click.option('--before-apt-commands', is_flag=True)
@click.option('--post-apt-commands', is_flag=True)
@click.option('--before-env-commands', is_flag=True)
@click.option('--post-env-commands', is_flag=True)
@click.option('--bootstrap', is_flag=True)
@click.option('--bootstrap-from', is_flag=True)
@click.option('--apt-packages', is_flag=True)
@click.option('--anaconda-version', is_flag=True)
@click.option('--python-version', is_flag=True)
@click.option('--env-name', is_flag=True)
@click.option('--env-create-arguments', is_flag=True)
@click.option('--sync-with', is_flag=True)
@click.option('--sync-target', is_flag=True)
@click.option('--exclude-from-sync', is_flag=True)
@click.option('--slurm-init-commands', is_flag=True)
@click.option('--singularity-init-commands', is_flag=True)
def get(ssh_username: bool = False,
        ssh_password: bool = False,
        ssh_host: bool = False,
        ssh_port: bool = False,
        recipe: bool = False,
        local_image: bool = False,
        remote_image: bool = False,
        before_apt_commands: bool = False,
        post_apt_commands: bool = False,
        before_env_commands: bool = False,
        post_env_commands: bool = False,
        bootstrap: bool = False,
        bootstrap_from: bool = False,
        apt_packages: bool = False,
        anaconda_version: bool = False,
        python_version: bool = False,
        env_name: bool = False,
        env_create_arguments: bool = False,
        sync_with: bool = False,
        sync_target: bool = False,
        exclude_from_sync: bool = False,
        slurm_init_commands: bool = False,
        singularity_init_commands: bool = False):
    """Read new parameters to the persistent experiment config by loading
    the config from the disk and accepting a set of indicators that
    specify which of the parameters to read to the terminal.

    Arguments:

    ssh_username: str
        a string representing the username of the account to use when
        connecting to the remote host using ssh.
    ssh_password: str
        a string representing the password of the account to use when
        connecting to the remote host using ssh.
    ssh_host: str
        a string representing the host name of the remote machine on
        which to upload a singularity image and launch jobs.
    ssh_port: int
        an integer representing the port on the remote machine to use
        when connecting to the machine to launch jobs.

    recipe: str
        the location on the disk to write a singularity recipe file
        which will be used later to build a singularity image.
    local_image: str
        the location on the disk to write a singularity image, which will
        be launched when running experiments.
    remote_image: str
        the location on the host to write a singularity image, which will
        be launched when running experiments.

    before_apt_commands: List[str]
        a list of commands to run while building the singularity image
        before apt packages are installed.
    post_apt_commands: List[str]
        a list of commands to run while building the singularity image
        after all apt packages have been installed.
    before_env_commands: List[str]
        a list of commands to run while building the singularity image
        before conda is downloaded and the env is created.
    post_env_commands: List[str]
        a list of commands to run while building the singularity image
        after conda is downloaded and the env has been created.

    bootstrap: str
        whether to bootstrap this singularity image from docker, and
        is set to 'docker' if this is the case.
    bootstrap_from: str
        the source to bootstrap from, which is the name of a docker
        container is bootstrapping from docker as above.
    apt_packages: List[str]
        a list of strings representing the names of apt packages to be
        installed in the current singularity image.

    anaconda_version: str
        the version number of the anaconda package to install, which
        can be set to 2021.05 as a simple default.
    python_version: str
        the version number of the python interpreter to install, which
        can be set to 3.7 as a simple default.
    env_name: str
        the name of the conda environment to build for this experiment
        which can simply be the name of the code-base.
    env_create_packages: str
        a string representing the names and channels of conda packages to
        be installed when creating the associated conda environment.

    sync_with: List[str]
        a string representing the path on disk where uncommitted code is
        stored and can be copied before starting experiments.
    sync_target: List[str]
        a string representing the path on disk where the code will be
        synced into, and experiments will be ran from.
    exclude_from_sync: List[str]
        a string representing the file pattern of files to exclude when
        synchronizing code with the singularity image.

    slurm_init_commands: List[str]
        a list of strings representing commands that are run within the
        slurm node before starting a singularity container.
    singularity_init_commands: List[str]
        a list of strings representing commands that are run within the
        singularity container before starting an experiment.

    """

    with PersistentExperimentConfig() as config:

        if ssh_username:
            print("ssh_username:", config.ssh_username)
        if ssh_password:
            print("ssh_password:", config.ssh_password)
        if ssh_host:
            print("ssh_host:", config.ssh_host)
        if ssh_port:
            print("ssh_port:", config.ssh_port)

        if recipe:
            print("recipe:", config.recipe)
        if local_image:
            print("local_image:", config.local_image)
        if remote_image:
            print("remote_image:", config.remote_image)

        if before_apt_commands:
            print("before_apt_commands:", config.before_apt_commands)
        if post_apt_commands:
            print("post_apt_commands:", config.post_apt_commands)
        if before_env_commands:
            print("before_env_commands:", config.before_env_commands)
        if post_env_commands:
            print("post_env_commands:", config.post_env_commands)

        if bootstrap:
            print("bootstrap:", config.bootstrap)
        if bootstrap_from:
            print("bootstrap_from:", config.bootstrap_from)
        if apt_packages:
            print("apt_packages:", config.apt_packages)

        if anaconda_version:
            print("anaconda_version:", config.anaconda_version)
        if python_version:
            print("python_version:", config.python_version)
        if env_name:
            print("env_name:", config.env_name)
        if env_create_arguments:
            print("env_create_arguments:", config.env_create_arguments)

        if sync_with:
            print("sync_with:", config.sync_with)
        if sync_target:
            print("sync_target:", config.sync_target)
        if exclude_from_sync:
            print("exclude_from_sync:", config.exclude_from_sync)

        if slurm_init_commands:
            print("slurm_init_commands:", config.slurm_init_commands)
        if singularity_init_commands:
            print("singularity_init_commands:",
                  config.singularity_init_commands)


@command_line_interface.command(
    context_settings=dict(ignore_unknown_options=True))
@click.option('--rebuild', is_flag=True)
@click.argument('commands', type=str, nargs=-1)
def local(rebuild: bool = False, commands: List[str] = ()):
    """Load the persistent experiment configuration file and launch an
    experiment locally by loading the singularity image and syncing local
    code with the code in the image and running commands.

    Arguments:

    rebuild: bool
        a boolean that controls whether the singularity image should be
        rebuilt even if it already exists on the disk.
    commands: List[str]
        a list of strings representing commands that are run within the
        container once all setup commands are finished.

    """

    with PersistentExperimentConfig() as config:
        commands = [" ".join(commands)] if len(commands) > 0 else []
        config.local_run(*commands, rebuild=rebuild)


@command_line_interface.command(
    context_settings=dict(ignore_unknown_options=True))
@click.option('--num-cpus', type=int, default=4)
@click.option('--num-gpus', type=int, default=1)
@click.option('--memory', type=int, default=16)
@click.option('--num-hours', type=int, default=8)
@click.option('--partition', type=str, default="russ_reserved")
@click.option('--exclude-nodes', type=str, default=None)
@click.option('--sweep-params', type=str, multiple=True, default=())
@click.option('--sweep-values', type=str, multiple=True, default=())
@click.option('--rebuild', is_flag=True)
@click.argument('commands', type=str, nargs=-1)
def remote(num_cpus: int = 4, num_gpus: int = 1, memory: int = 16, num_hours: int = 8,
           partition: str = "russ_reserved", exclude_nodes: str = None,
           sweep_params: List[str] = (), sweep_values: List[str] = (),
           rebuild: bool = False, commands: List[str] = ()):
    """Load the persistent experiment configuration file and launch an
    experiment remotely by loading the singularity image and syncing local
    code with the code in the remote image and running commands.

    Arguments:

    num_cpus: int
        an integer representing the number of cpu cores that will be
        allocated by slurm to the generated slurm job.
    num_gpus: int
        an integer representing the number of gpu nodes that will be
        allocated by slurm to the generated slurm job.
    memory: int
        an integer representing the amount of memory that will be
        allocated by slurm to the generated slurm job.
    num_hours: int
        an integer representing the amount of time the slurm job will be
        allowed to run before forcibly terminating.
    partition: str
        a string that represents the slurm partition of machines to use
        when scheduling a slurm job on the host machine.

    sweep_params: List[str]
        a list of strings representing names of a grid of parameters of the
        specified command that will be searched and replaced.
    sweep_values: List[str]
        a list of strings representing values of a grid of parameters of the
        specified command that will be searched and replaced.

    rebuild: bool
        a boolean that controls whether the singularity image should be
        rebuilt even if it already exists on the disk.
    commands: List[str]
        a list of strings representing commands that are run within the
        container once all setup commands are finished.

    """

    with PersistentExperimentConfig() as config:
        commands = [" ".join(commands)] if len(commands) > 0 else []
        config.remote_run(*commands, rebuild=rebuild, partition=partition,
                          num_cpus=num_cpus, num_gpus=num_gpus, memory=memory,
                          num_hours=num_hours, exclude_nodes=exclude_nodes,
                          sweep_params=sweep_params, sweep_values=sweep_values)


@command_line_interface.command(
    context_settings=dict(ignore_unknown_options=True))
@click.option('--interval', type=float, default=1.0)
@click.option('--watch', is_flag=True)
@click.argument('commands', type=str, nargs=-1)
def shell(interval: float = 1.0,
          watch: bool = False, commands: List[str] = ()):
    """Run a set of bash commands on the remote machine, which can be used to
    check on jobs that are scheduled or running on the host, and also to
    cancel jobs that are scheduled or running.

    Arguments:

    commands: List[str]
        a list of strings representing commands that are run on the host
        machine through an ssh connecting.
    watch: bool
        a boolean that controls whether the remote commands should be
        executed repeatedly at a specified interval.
    interval: float
        a float that represents the amount of time in seconds between
        successive commands when watch is set to True.

    """

    with PersistentExperimentConfig() as config:
        config.remote_shell(" ".join(commands),
                            watch=watch, interval=interval)


@command_line_interface.command()
@click.option('--recursive', is_flag=True)
@click.option('--exclude', type=str, default="")
@click.argument('source-path', type=str, nargs=1)
@click.argument('destination-path', type=str, nargs=1)
def upload(recursive: bool = False, exclude: str = "",
           source_path: str = "", destination_path: str = ""):
    """Load the persistent experiment configuration file and start copying
    files from the source location on the local disk to the destination
    path which is inside the singularity image on the host.

    Arguments:

    source_path: str
        a string representing the path on the local disk to a file or
        directory that will be copied from.
    destination_path: str
        a string representing the path on the remote disk to a file or
        directory that will be copied into from a source.
    recursive: bool
        a boolean that controls whether rsync will be called with the
        recursive option to copy a directory.
    exclude: str
        a string representing the file pattern for files to exclude
        from the copy, such as data files.

    """

    with PersistentExperimentConfig() as config:
        config.remote_rsync(source_path, os.path.join(
            config.remote_image, destination_path),
                            recursive=recursive, exclude=exclude,
                            source_is_remote=False, destination_is_remote=True)


@command_line_interface.command()
@click.option('--recursive', is_flag=True)
@click.option('--exclude', type=str, default="")
@click.argument('source-path', type=str, nargs=1)
@click.argument('destination-path', type=str, nargs=1)
def download(recursive: bool = False, exclude: str = "",
             source_path: str = "", destination_path: str = ""):
    """Load the persistent experiment configuration file and start copying
    files from the source location in the singularity image on the remote
    disk to the destination path which is on the local disk

    Arguments:

    source_path: str
        a string representing the path on the remote disk to a file or
        directory that will be copied from.
    destination_path: str
        a string representing the path on the local disk to a file or
        directory that will be copied into from a source.
    recursive: bool
        a boolean that controls whether rsync will be called with the
        recursive option to copy a directory.
    exclude: str
        a string representing the file pattern for files to exclude
        from the copy, such as data files.

    """

    with PersistentExperimentConfig() as config:
        config.remote_rsync(os.path.join(config.remote_image, source_path),
                            destination_path, recursive=recursive,
                            exclude=exclude, source_is_remote=True,
                            destination_is_remote=False)


@command_line_interface.command()
@click.argument('file', type=str, nargs=1)
def dump(file: str = None):
    """Export the current configuration parameters, excluding the ssh
    credentials to a file as specified in the command line argument,
    or simply by printing to the terminal.

    Arguments:

    file: str
        a string representing the path on the local disk to export a file
        containing the current persistent experiment configuration.

    """

    # export a dictionary containing the non private configuration info
    with PersistentExperimentConfig() as config:
        with open(file, "w") as f:
            json.dump(dict(recipe=config.recipe,
                           local_image=config.local_image,
                           remote_image=config.remote_image,

                           before_apt_commands=config.before_apt_commands,
                           post_apt_commands=config.post_apt_commands,
                           before_env_commands=config.before_env_commands,
                           post_env_commands=config.post_env_commands,

                           bootstrap=config.bootstrap,
                           bootstrap_from=config.bootstrap_from,
                           apt_packages=config.apt_packages,

                           anaconda_version=config.anaconda_version,
                           python_version=config.python_version,
                           env_name=config.env_name,
                           env_create_arguments=config.env_create_arguments,

                           sync_with=config.sync_with,
                           sync_target=config.sync_target,
                           exclude_from_sync=config.exclude_from_sync,

                           slurm_init_commands=config.slurm_init_commands,
                           singularity_init_commands=
                           config.singularity_init_commands), f, indent=4)


@command_line_interface.command()
@click.argument('file', type=str, nargs=1)
def load(file: str = None):
    """Load the existing configuration parameters, excluding the ssh
    credentials from a file as specified in the command line argument,
    overwriting the current persistent configuration parameters.

    Arguments:

    file: str
        a string representing the path on the local disk to load a file
        containing the new persistent experiment configuration.

    """

    with open(file, "r") as f:
        data = json.load(f)  # the format of this object is a dictionary

    # load a dictionary containing non private configuration info
    with PersistentExperimentConfig() as config:
        config.recipe = data["recipe"]
        config.local_image = data["local_image"]
        config.remote_image = data["remote_image"]

        config.before_apt_commands = data["before_apt_commands"]
        config.post_apt_commands = data["post_apt_commands"]
        config.before_env_commands = data["before_env_commands"]
        config.post_env_commands = data["post_env_commands"]

        config.bootstrap = data["bootstrap"]
        config.bootstrap_from = data["bootstrap_from"]
        config.apt_packages = data["apt_packages"]

        config.anaconda_version = data["anaconda_version"]
        config.python_version = data["python_version"]
        config.env_name = data["env_name"]
        config.env_create_arguments = data["env_create_arguments"]

        config.sync_with = data["sync_with"]
        config.sync_target = data["sync_target"]
        config.exclude_from_sync = data["exclude_from_sync"]

        config.slurm_init_commands = data["slurm_init_commands"]
        config.singularity_init_commands = data["singularity_init_commands"]


@command_line_interface.command()
def clear():
    """Clear the current experiment configuration and write a default one
    to the disk in place of the existing persistent configuration file,
    which can be teh fastest way to remove undesired states.

    """

    # load a dictionary containing non private configuration info
    with PersistentExperimentConfig(clear=True):
        pass  # do nothing and write the default config


if __name__ == "__main__":
    command_line_interface()  # expose a public command line interface
