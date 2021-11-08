import time
import sys
import os
import paramiko
import pickle as pkl
import click


from pexpect import spawn, EOF
from typing import List


DEFAULT_RECIPE = r"""Bootstrap: {bootstrap}
From: {bootstrap_from}

%post

#Add nvidia driver paths to the environment variables
echo "\n #Nvidia driver paths \n" >> /environment
echo 'export PATH="/nvbin:$PATH"' >> /environment
echo 'export LD_LIBRARY_PATH="/nvlib:$LD_LIBRARY_PATH"' >> /environment

#Add CUDA paths
echo "\n #Cuda paths \n" >> /environment
echo 'export CPATH="/usr/local/cuda/include:$CPATH"' >> /environment
echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> /environment
echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> /environment
echo 'export CUDA_HOME="/usr/local/cuda"' >> /environment

touch /bin/nvidia-smi
touch /usr/bin/nvidia-smi
touch /usr/bin/nvidia-debugdump
touch /usr/bin/nvidia-persistenced
touch /usr/bin/nvidia-cuda-mps-control
touch /usr/bin/nvidia-cuda-mps-server

mkdir /etc/dcv
mkdir /var/lib/dcv-gl
mkdir /usr/lib64

apt-get update -y
{apt_packages}

wget https://repo.anaconda.com/archive/Anaconda3-{anaconda_version}-Linux-x86_64.sh -O /anaconda3.sh
bash /anaconda3.sh -b -p /anaconda3
rm /anaconda3.sh

. /anaconda3/etc/profile.d/conda.sh

mkdir /code
mkdir /results

conda create -y -n {env_name} python={python_version} {env_packages}
conda activate {env_name}

git clone {git_url} {git_target}
{install_command}

chmod -R 777 /code
chmod -R 777 /results
chmod -R 777 /anaconda3"""


# arguments for setting up the package environment in singularity
DEFAULT_BOOTSTRAP = "docker"
DEFAULT_BOOTSTRAP_FROM = "nvidia/cuda:11.3.1-cudnn8-devel-ubuntu16.04"
DEFAULT_APT_PACKAGES = ("unzip", "htop", "wget",
                        "git", "vim", "build-essential")


# arguments that describe the conda environment to build
DEFAULT_ANACONDA_VERSION = "2021.05"
DEFAULT_PYTHON_VERSION = "3.7"
DEFAULT_ENV_NAME = "nerf"
DEFAULT_ENV_PACKAGES = "pytorch torchvision " \
                       "torchaudio cudatoolkit=11.3 -c pytorch"


# a default location for the singularity image and singularity recipe
DEFAULT_LOCAL_RECIPE = "experiment.recipe"
DEFAULT_LOCAL_IMAGE = "experiment.sif"
DEFAULT_REMOTE_RECIPE = "experiment.recipe"
DEFAULT_REMOTE_IMAGE = "experiment.sif"


# arguments that describe how to install the experiment code from github
DEFAULT_GIT_URL = "https://github.com/brandontrabucco/nerf.git"
DEFAULT_GIT_TARGET = "/code/nerf"
DEFAULT_INSTALL_COMMAND = "pip install -e {git_target}"


# information about how to sync code before running an experiment
DEFAULT_SYNC = True
DEFAULT_SYNC_WITH = "/home/btrabucco/PycharmProjects/nerf"
DEFAULT_EXCLUDE_FROM_SYNC = "*.pkl"


# a default command that may be run in the container
DEFAULT_INIT_COMMANDS = ("sleep 1",
                         "echo 'running test experiment'")


# a template for running experiment commands in the container
SINGULARITY_EXEC_TEMPLATE = "singularity \
    exec --nv -w {image} bash -c \"{singularity_command}\""


# a template for launching an experiment using a slurm scheduler
SLURM_SRUN_TEMPLATE = "srun --cpus-per-task={num_cpus} \
    --gres=gpu:{num_gpus} --mem={memory}g \
    --time={num_hours}:00:00 -p russ_reserved {slurm_command}"


# credentials for logging in to the remote host using ssh
DEFAULT_SSH_USERNAME = "username"
DEFAULT_SSH_PASSWORD = "password"
DEFAULT_SSH_HOST = "matrix.ml.cmu.edu"
DEFAULT_SSH_PORT = 22
DEFAULT_SSH_SLEEP_SECONDS = 0.001


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

    local_recipe: str
        the location on the disk to write a singularity recipe file
        which will be used later to build a singularity image.
    local_image: str
        the location on the disk to write a singularity image, which will
        be launched when running experiments.
    remote_recipe: str
        the location on the host to write a singularity recipe file
        which will be used later to build a singularity image.
    remote_image: str
        the location on the host to write a singularity image, which will
        be launched when running experiments.

    env_name: str
        the name of the conda environment to build for this experiment
        which can simply be the name of the code-base.
    env_packages: str
        a string representing the names and channels of conda packages to
        be installed when creating the associated conda environment.

    git_url: str
        a string representing the url where the experiment code is
        available for download using a git clone command.
    git_target: str
        a string representing the path on disk where the code will be
        cloned into, and experiments will be ran from.
    install_command: str
        a string that instructs singularity how to install the experiment
        code, which can be as simple as a pip or conda install.

    sync: bool
        a boolean that controls whether to sync the contents of the local
        code working directory to the singularity image.
    sync_with: str
        a string representing the path on disk where uncommitted code is
        stored and can be copied before starting experiments.
    exclude_from_sync: str
        a string representing the file pattern of files to exclude when
        synchronizing code with the singularity image.

    init_commands: List[str]
        a list of strings representing commands that are run within the
        container before starting an experiment.

    """

    def __init__(self, ssh_username: str = DEFAULT_SSH_USERNAME,
                 ssh_password: str = DEFAULT_SSH_PASSWORD,
                 ssh_host: str = DEFAULT_SSH_HOST,
                 ssh_port: int = DEFAULT_SSH_PORT,
                 bootstrap: str = DEFAULT_BOOTSTRAP,
                 bootstrap_from: str = DEFAULT_BOOTSTRAP_FROM,
                 apt_packages: List[str] = DEFAULT_APT_PACKAGES,
                 anaconda_version: str = DEFAULT_ANACONDA_VERSION,
                 python_version: str = DEFAULT_PYTHON_VERSION,
                 local_recipe: str = DEFAULT_LOCAL_RECIPE,
                 local_image: str = DEFAULT_LOCAL_IMAGE,
                 remote_recipe: str = DEFAULT_REMOTE_RECIPE,
                 remote_image: str = DEFAULT_REMOTE_IMAGE,
                 env_name: str = DEFAULT_ENV_NAME,
                 env_packages: str = DEFAULT_ENV_PACKAGES,
                 git_url: str = DEFAULT_GIT_URL,
                 git_target: str = DEFAULT_GIT_TARGET,
                 install_command: str = DEFAULT_INSTALL_COMMAND,
                 sync: bool = DEFAULT_SYNC,
                 sync_with: str = DEFAULT_SYNC_WITH,
                 exclude_from_sync: str = DEFAULT_EXCLUDE_FROM_SYNC,
                 init_commands: List[str] = DEFAULT_INIT_COMMANDS):
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

        local_recipe: str
            the location on the disk to write a singularity recipe file
            which will be used later to build a singularity image.
        local_image: str
            the location on the disk to write a singularity image, which will
            be launched when running experiments.
        remote_recipe: str
            the location on the host to write a singularity recipe file
            which will be used later to build a singularity image.
        remote_image: str
            the location on the host to write a singularity image, which will
            be launched when running experiments.

        env_name: str
            the name of the conda environment to build for this experiment
            which can simply be the name of the code-base.
        env_packages: str
            a string representing the names and channels of conda packages to
            be installed when creating the associated conda environment.

        git_url: str
            a string representing the url where the experiment code is
            available for download using a git clone command.
        git_target: str
            a string representing the path on disk where the code will be
            cloned into, and experiments will be ran from.
        install_command: str
            a string that instructs singularity how to install the experiment
            code, which can be as simple as a pip or conda install.

        sync: bool
            a boolean that controls whether to sync the contents of the local
            code working directory to the singularity image.
        sync_with: str
            a string representing the path on disk where uncommitted code is
            stored and can be copied before starting experiments.
        exclude_from_sync: str
            a string representing the file pattern of files to exclude when
            synchronizing code with the singularity image.

        init_commands: List[str]
            a list of strings representing commands that are run within the
            container before starting an experiment.

        """

        # arguments for the ssh login credentials of the host
        self.ssh_username = ssh_username
        self.ssh_password = ssh_password
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port

        # global arguments for the singularity image package environment
        self.bootstrap = bootstrap
        self.bootstrap_from = bootstrap_from
        self.apt_packages = apt_packages
        self.anaconda_version = anaconda_version
        self.python_version = python_version

        # locations for a singularity recipe and image to be written
        self.local_recipe = local_recipe
        self.local_image = os.path.realpath(local_image)
        self.remote_recipe = remote_recipe
        self.remote_image = remote_image

        # arguments that specify the package environment for the source code
        self.env_name = env_name
        self.env_packages = env_packages

        # arguments that specify where and how to install source code
        self.git_url = git_url
        self.git_target = git_target
        self.install_command = install_command.format(git_target=git_target)

        # arguments that specify how to sync code before an experiment
        self.sync = sync
        self.sync_with = sync_with
        self.exclude_from_sync = exclude_from_sync

        # always initialize conda before running experiments, and then
        # run any user provided code afterwards
        init_commands = [command.format(
            git_target=git_target) for command in init_commands]
        self.init_commands = [". /anaconda3/etc/profile.d/conda.sh",
                              "conda activate {}".format(env_name),
                              "cd {}".format(git_target), *init_commands]

    def local_recipe_exists(self) -> bool:
        """Utility function that checks the local disk for whether a
        singularity recipe with the given name already exists on the disk
        at the desired location, and if so returns true.

        Returns:

        recipe_exists: bool
            a boolean that returns True if the singularity recipe with the
            specified name already exists on the disk.

        """

        return os.path.exists(self.local_recipe)  # exists at this location

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

    def remote_recipe_exists(self) -> bool:
        """Utility function that checks the remote host for whether a
        singularity recipe with the given name already exists on the host
        at the desired location, and if so returns true.

        Returns:

        recipe_exists: bool
            a boolean that returns True if the singularity image with the
            specified name already exists on the host.

        """

        # open an ssh connection to the remote host by logging in using the
        # provided username and password for that machine
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(self.ssh_host, self.ssh_port,
                       username=self.ssh_username, password=self.ssh_password,
                       look_for_keys=False, allow_agent=False)

        # open an sftp client and check if a file exists on the remote host
        with client.open_sftp() as sftp:
            return self.remote_path_exists(sftp, self.remote_recipe)

    def remote_image_exists(self) -> bool:
        """Utility function that checks the remote host for whether a
        singularity image with the given name already exists on the host
        at the desired location, and if so returns true.

        Returns:

        image_exists: bool
            a boolean that returns True if the singularity image with the
            specified name already exists on the host.

        """

        # open an ssh connection to the remote host by logging in using the
        # provided username and password for that machine
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(self.ssh_host, self.ssh_port,
                       username=self.ssh_username, password=self.ssh_password,
                       look_for_keys=False, allow_agent=False)

        # open an sftp client and check if a file exists on the remote host
        with client.open_sftp() as sftp:
            return self.remote_path_exists(sftp, self.remote_image)

    def write_singularity_recipe(self):
        """Using the provided class attributes, write a singularity recipe
        to the disk, which will be used in a later stage to build a
        singularity image for performing experiments.

        """

        with open(self.local_recipe, "w") as recipe_file:
            recipe_file.write(DEFAULT_RECIPE.format(
                bootstrap=self.bootstrap,
                bootstrap_from=self.bootstrap_from,
                anaconda_version=self.anaconda_version,
                python_version=self.python_version,
                env_name=self.env_name,
                env_packages=self.env_packages,
                git_url=self.git_url,
                git_target=self.git_target,
                install_command=self.install_command,
                apt_packages="\n".join(["apt-get install -y {package}"
                                       .format(package=p)
                                        for p in self.apt_packages])))

    def write_singularity_image(self, **kwargs):
        """Using the provided class attributes, generate a singularity
        recipe file and build a singularity image that will be used to run
        experiments in an isolated package environment.

        """

        # if the recipe does not exist locally then write it first
        if not self.local_recipe_exists():
            self.write_singularity_recipe()

        # build the singularity image using the singularity api
        from spython.main import Client
        Client.build(recipe=self.local_recipe, image=self.local_image,
                     sudo=False, sandbox=True,
                     options=["--fakeroot"], **kwargs)

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

        # spawn a child process that copies files to the host using rsync
        child = spawn("rsync", ["-ra" if recursive else "-a", "--progress",
                                "--exclude", exclude,
                                source_path,
                                destination_path], encoding='utf-8')

        # print outputs of the process to the terminal and wait for the
        # process to finish copying files in the tree
        child.logfile = sys.stdout
        child.expect(EOF, timeout=10800)  # catch when it finishes with EOF

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

        # spawn a child process that copies files to the host using rsync
        child = spawn("rsync", ["-ra" if recursive else "-a", "--progress",
                                "--exclude", exclude,
                                "-e", "ssh -p {}".format(self.ssh_port),
                                source_path,
                                destination_path], encoding='utf-8')

        # print outputs of the process to the terminal
        child.logfile = sys.stdout

        # expect the host to prompt our client for a kuberos password
        child.expect("{username}@{host}'s password:"
                     .format(username=self.ssh_username, host=self.ssh_host))

        # once we have been prompted for a password enter it into the stdin
        # and wait until the child process finishes
        time.sleep(DEFAULT_SSH_SLEEP_SECONDS)
        child.sendline(self.ssh_password)
        child.expect(EOF, timeout=10800)  # catch when it finishes with EOF

    def upload_singularity_recipe(self):
        """Using the provided class attributes, generate and run a command in
        a bash shell that will write a singularity recipe and copy it
        from the local disk to a remote host machine.

        """

        # if the recipe does not exist locally then write it first
        if not self.local_recipe_exists():
            self.write_singularity_recipe()

        # copy the singularity recipe file to the host
        self.remote_rsync(os.path.join(self.local_recipe, "."),
                          self.remote_recipe, source_is_remote=False,
                          destination_is_remote=True, recursive=False)

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

    def run_in_singularity(self, *commands: List[str],
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

        return SINGULARITY_EXEC_TEMPLATE.format(
            singularity_command=" && ".join(
                self.init_commands + list(commands)), image=image)

    def local_run(self, *commands: List[str],
                  sync: bool = None, rebuild: bool = False):
        """Generate and run a command in the bash shell that starts a
        singularity container locally and runs commands in that container
        and prints outputs to the standard output stream.

        Arguments:

        commands: List[str]
            a list of strings representing commands that are run within the
            container once all setup commands are finished.
        sync: bool
            a boolean that controls whether to sync the contents of the local
            code working directory to the singularity image.
        rebuild: bool
            a boolean that controls whether the singularity image should be
            rebuilt even if it already exists on the disk.

        """

        # if the recipe does not exist locally then write it first
        if not self.local_image_exists() or rebuild:
            self.write_singularity_image()  # build the singularity image

        # copy the local code directory to the local singularity image
        if sync if sync is not None else self.sync:
            self.local_rsync(os.path.join(self.sync_with, "."), os.path.join(
                self.local_image, self.git_target[1:]),
                recursive=True, exclude=self.exclude_from_sync)

        # start an experiment locally using a local singularity container
        stdout = os.popen(self.run_in_singularity(*commands,
                                                  image=self.local_image))

        # print the output from the terminal as the command runs
        for line in iter(stdout.readline, ""):
            print(line)  # prints even if the command is not yet finished

    def run_in_slurm(self, *commands: List[str], num_cpus: int = 4,
                     num_gpus: int = 1, memory: int = 16, num_hours: int = 8,
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

        Returns:

        run_command: str
            a string representing a command that can be executed in the
            terminal in order to run experiments using singularity.

        """

        return SLURM_SRUN_TEMPLATE.format(
            num_cpus=num_cpus, num_gpus=num_gpus, num_hours=num_hours,
            memory=memory, slurm_command=self.run_in_singularity(*commands,
                                                                 image=image))

    def remote_run(self, *commands: List[str], sync: bool = None,
                   rebuild: bool = False, num_cpus: int = 4,
                   num_gpus: int = 1, memory: int = 16, num_hours: int = 8):
        """Generate and run a command in the bash shell that starts a
        singularity container remotely and runs commands in that container
        and prints outputs to the standard output stream.

        Arguments:

        commands: List[str]
            a list of strings representing commands that are run within the
            container once all setup commands are finished.
        sync: bool
            a boolean that controls whether to sync the contents of the local
            code working directory to the singularity image.
        rebuild: bool
            a boolean that controls whether the singularity image should be
            rebuilt even if it already exists on the disk.

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

        """

        # if the image does not exist on the remote then upload it first
        if not self.remote_image_exists() or rebuild:
            self.upload_singularity_image()  # build the singularity image

        # copy the local code directory to the remote singularity image
        if sync if sync is not None else self.sync:
            self.remote_rsync(os.path.join(self.sync_with, "."), os.path.join(
                self.remote_image, self.git_target[1:]),
                recursive=True, exclude=self.exclude_from_sync,
                source_is_remote=False, destination_is_remote=True)

        # open an ssh connection to the remote host by logging in using the
        # provided username and password for that machine
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(self.ssh_host, self.ssh_port,
                       username=self.ssh_username, password=self.ssh_password,
                       look_for_keys=False, allow_agent=False)

        # generate a command to launch a remote experiment using slurm
        stdout = client.exec_command(self.run_in_slurm(
            *commands, image=self.remote_image,
            num_cpus=num_cpus, num_gpus=num_gpus,
            num_hours=num_hours, memory=memory), get_pty=True)[1]

        # print the output from the terminal as the command runs
        for line in iter(stdout.readline, ""):
            print(line)  # prints even if the command is not yet finished


# the default location for a config file to be stored on the local disk
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.pkl")


class PersistentExperimentConfig(object):
    """Create a persistent wrapper around the ExperimentConfig class that
    enables saving and loading the config multiple times when the
    parameters are changed and experiments are launched.

    """

    def __init__(self, storage_path: str = DEFAULT_CONFIG, **kwargs):
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
        if not os.path.exists(self.storage_path):
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
@click.option('--bootstrap', type=str, default=None)
@click.option('--bootstrap-from', type=str, default=None)
@click.option('--apt-packages', type=str, default=None, multiple=True)
@click.option('--anaconda-version', type=str, default=None)
@click.option('--python-version', type=str, default=None)
@click.option('--local-recipe', type=str, default=None)
@click.option('--local-image', type=str, default=None)
@click.option('--remote-recipe', type=str, default=None)
@click.option('--remote-image', type=str, default=None)
@click.option('--env-name', type=str, default=None)
@click.option('--env-packages', type=str, default=None)
@click.option('--git-url', type=str, default=None)
@click.option('--git-target', type=str, default=None)
@click.option('--install-command', type=str, default=None)
@click.option('--sync', type=bool, default=None)
@click.option('--sync-with', type=str, default=None)
@click.option('--exclude-from-sync', type=str, default=None)
@click.option('--init-commands', type=str, default=None, multiple=True)
def set(ssh_username: str = DEFAULT_SSH_USERNAME,
        ssh_password: str = DEFAULT_SSH_PASSWORD,
        ssh_host: str = DEFAULT_SSH_HOST,
        ssh_port: int = DEFAULT_SSH_PORT,
        bootstrap: str = DEFAULT_BOOTSTRAP,
        bootstrap_from: str = DEFAULT_BOOTSTRAP_FROM,
        apt_packages: List[str] = DEFAULT_APT_PACKAGES,
        anaconda_version: str = DEFAULT_ANACONDA_VERSION,
        python_version: str = DEFAULT_PYTHON_VERSION,
        local_recipe: str = DEFAULT_LOCAL_RECIPE,
        local_image: str = DEFAULT_LOCAL_IMAGE,
        remote_recipe: str = DEFAULT_REMOTE_RECIPE,
        remote_image: str = DEFAULT_REMOTE_IMAGE,
        env_name: str = DEFAULT_ENV_NAME,
        env_packages: str = DEFAULT_ENV_PACKAGES,
        git_url: str = DEFAULT_GIT_URL,
        git_target: str = DEFAULT_GIT_TARGET,
        install_command: str = DEFAULT_INSTALL_COMMAND,
        sync: bool = DEFAULT_SYNC,
        sync_with: str = DEFAULT_SYNC_WITH,
        exclude_from_sync: str = DEFAULT_EXCLUDE_FROM_SYNC,
        init_commands: List[str] = DEFAULT_INIT_COMMANDS):
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

    local_recipe: str
        the location on the disk to write a singularity recipe file
        which will be used later to build a singularity image.
    local_image: str
        the location on the disk to write a singularity image, which will
        be launched when running experiments.
    remote_recipe: str
        the location on the host to write a singularity recipe file
        which will be used later to build a singularity image.
    remote_image: str
        the location on the host to write a singularity image, which will
        be launched when running experiments.

    env_name: str
        the name of the conda environment to build for this experiment
        which can simply be the name of the code-base.
    env_packages: str
        a string representing the names and channels of conda packages to
        be installed when creating the associated conda environment.

    git_url: str
        a string representing the url where the experiment code is
        available for download using a git clone command.
    git_target: str
        a string representing the path on disk where the code will be
        cloned into, and experiments will be ran from.
    install_command: str
        a string that instructs singularity how to install the experiment
        code, which can be as simple as a pip or conda install.

    sync: bool
        a boolean that controls whether to sync the contents of the local
        code working directory to the singularity image.
    sync_with: str
        a string representing the path on disk where uncommitted code is
        stored and can be copied before starting experiments.
    exclude_from_sync: str
        a string representing the file pattern of files to exclude when
        synchronizing code with the singularity image.

    init_commands: List[str]
        a list of strings representing commands that are run within the
        container before starting an experiment.

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

        if bootstrap is not None:
            config.bootstrap = bootstrap
        if bootstrap_from is not None:
            config.bootstrap_from = bootstrap_from

        if apt_packages is not None and len(apt_packages) > 0:
            config.apt_packages = apt_packages
        if anaconda_version is not None:
            config.anaconda_version = anaconda_version
        if python_version is not None:
            config.python_version = python_version

        if local_recipe is not None:
            config.local_recipe = local_recipe
        if local_image is not None:
            config.local_image = os.path.realpath(local_image)
        if remote_recipe is not None:
            config.remote_recipe = remote_recipe
        if remote_image is not None:
            config.remote_image = remote_image

        if env_name is not None:
            config.env_name = env_name
        if env_packages is not None:
            config.env_packages = env_packages

        if git_url is not None:
            config.git_url = git_url
        if git_target is not None:
            config.git_target = git_target
        if install_command is not None:
            config.install_command = \
                install_command.format(git_target=config.git_target)

        if sync is not None:
            config.sync = sync
        if sync_with is not None:
            config.sync_with = sync_with
        if exclude_from_sync is not None:
            config.exclude_from_sync = exclude_from_sync

        if init_commands is not None and len(init_commands) > 0:
            init_commands = [command.format(
                git_target=config.git_target) for command in init_commands]
            config.init_commands = [
                ". /anaconda3/etc/profile.d/conda.sh",
                "conda activate {}".format(config.env_name),
                "cd {}".format(config.git_target), *init_commands]


@command_line_interface.command()
@click.option('--ssh-username', is_flag=True)
@click.option('--ssh-password', is_flag=True)
@click.option('--ssh-host', is_flag=True)
@click.option('--ssh-port', is_flag=True)
@click.option('--bootstrap', is_flag=True)
@click.option('--bootstrap-from', is_flag=True)
@click.option('--apt-packages', is_flag=True)
@click.option('--anaconda-version', is_flag=True)
@click.option('--python-version', is_flag=True)
@click.option('--local-recipe', is_flag=True)
@click.option('--local-image', is_flag=True)
@click.option('--remote-recipe', is_flag=True)
@click.option('--remote-image', is_flag=True)
@click.option('--env-name', is_flag=True)
@click.option('--env-packages', is_flag=True)
@click.option('--git-url', is_flag=True)
@click.option('--git-target', is_flag=True)
@click.option('--install-command', is_flag=True)
@click.option('--sync', is_flag=True)
@click.option('--sync-with', is_flag=True)
@click.option('--exclude-from-sync', is_flag=True)
@click.option('--init-commands', is_flag=True)
def get(ssh_username: bool = False,
        ssh_password: bool = False,
        ssh_host: bool = False,
        ssh_port: bool = False,
        bootstrap: bool = False,
        bootstrap_from: bool = False,
        apt_packages: bool = False,
        anaconda_version: bool = False,
        python_version: bool = False,
        local_recipe: bool = False,
        local_image: bool = False,
        remote_recipe: bool = False,
        remote_image: bool = False,
        env_name: bool = False,
        env_packages: bool = False,
        git_url: bool = False,
        git_target: bool = False,
        install_command: bool = False,
        sync: bool = False,
        sync_with: bool = False,
        exclude_from_sync: bool = False,
        init_commands: bool = False):
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

    local_recipe: str
        the location on the disk to write a singularity recipe file
        which will be used later to build a singularity image.
    local_image: str
        the location on the disk to write a singularity image, which will
        be launched when running experiments.
    remote_recipe: str
        the location on the host to write a singularity recipe file
        which will be used later to build a singularity image.
    remote_image: str
        the location on the host to write a singularity image, which will
        be launched when running experiments.

    env_name: str
        the name of the conda environment to build for this experiment
        which can simply be the name of the code-base.
    env_packages: str
        a string representing the names and channels of conda packages to
        be installed when creating the associated conda environment.

    git_url: str
        a string representing the url where the experiment code is
        available for download using a git clone command.
    git_target: str
        a string representing the path on disk where the code will be
        cloned into, and experiments will be ran from.
    install_command: str
        a string that instructs singularity how to install the experiment
        code, which can be as simple as a pip or conda install.

    sync: bool
        a boolean that controls whether to sync the contents of the local
        code working directory to the singularity image.
    sync_with: str
        a string representing the path on disk where uncommitted code is
        stored and can be copied before starting experiments.
    exclude_from_sync: str
        a string representing the file pattern of files to exclude when
        synchronizing code with the singularity image.

    init_commands: List[str]
        a list of strings representing commands that are run within the
        container before starting an experiment.

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

        if local_recipe:
            print("local_recipe:", config.local_recipe)
        if local_image:
            print("local_image:", config.local_image)
        if remote_recipe:
            print("remote_recipe:", config.remote_recipe)
        if remote_image:
            print("remote_image:", config.remote_image)

        if env_name:
            print("env_name:", config.env_name)
        if env_packages:
            print("env_packages:", config.env_packages)

        if git_url:
            print("git_url:", config.git_url)
        if git_target:
            print("git_target:", config.git_target)
        if install_command:
            print("install_command:", config.install_command)

        if sync:
            print("sync:", config.sync)
        if sync_with:
            print("sync_with:", config.sync_with)
        if exclude_from_sync:
            print("exclude_from_sync:", config.exclude_from_sync)

        if init_commands:
            print("init_commands:", config.init_commands)


@command_line_interface.command()
@click.option('--rebuild', is_flag=True)
@click.option('--sync', is_flag=True)
@click.argument('commands', type=str, nargs=-1)
def local(rebuild: bool = False,
          sync: bool = False, commands: List[str] = ()):
    """Load the persistent experiment configuration file and launch an
    experiment locally by loading the singularity image and syncing local
    code with the code in the image and running commands.

    Arguments:

    rebuild: bool
        a boolean that controls whether the singularity image should be
        rebuilt even if it already exists on the disk.
    sync: bool
        a boolean that controls whether to sync the contents of the local
        code working directory to the singularity image.
    commands: List[str]
        a list of strings representing commands that are run within the
        container once all setup commands are finished.

    """

    with PersistentExperimentConfig() as config:
        config.local_run(" ".join(commands), sync=sync, rebuild=rebuild)


@command_line_interface.command()
@click.option('--num-cpus', type=int, default=4)
@click.option('--num-gpus', type=int, default=1)
@click.option('--memory', type=int, default=16)
@click.option('--num-hours', type=int, default=8)
@click.option('--rebuild', is_flag=True)
@click.option('--sync', is_flag=True)
@click.argument('commands', type=str, nargs=-1)
def remote(num_cpus: int = 4, num_gpus: int = 1,
           memory: int = 16, num_hours: int = 8,
           rebuild: bool = False,
           sync: bool = False, commands: List[str] = ()):
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

    rebuild: bool
        a boolean that controls whether the singularity image should be
        rebuilt even if it already exists on the disk.
    sync: bool
        a boolean that controls whether to sync the contents of the local
        code working directory to the singularity image.
    commands: List[str]
        a list of strings representing commands that are run within the
        container once all setup commands are finished.

    """

    with PersistentExperimentConfig() as config:
        config.remote_run(" ".join(commands), sync=sync, rebuild=rebuild,
                          num_cpus=num_cpus, num_gpus=num_gpus,
                          memory=memory, num_hours=num_hours)


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


if __name__ == "__main__":
    command_line_interface()  # expose a public command line interface
