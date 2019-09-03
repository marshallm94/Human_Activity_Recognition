#!/bin/bash

sudo yum -y install tmux

# creating a anaconda directory in the home directory
sudo mkdir $HOME/anaconda
sudo chown ec2-user:ec2-user $HOME/anaconda

# Downloading Anaconda installation script
# wget -S option = server-response (print server response)
# wget -T option = timeout (set all timeout values to seconds)
# wget -t option = tries (set number of retries)
# wget -O option = output-document (write documents to a file)
wget -S -T 10 -t 5 https://repo.continuum.io/archive/Anaconda3-2019.07-Linux-x86_64.sh -O $HOME/anaconda/anaconda_install.sh

# running Anaconda installation script
# anaconda_install -b option = install in BATCH mode. Assumes you agree to the license agreement
# anaconda_install -p option = install prefix/path
# anaconda_install -u option = update
bash $HOME/anaconda/anaconda_install.sh -u -b -p $HOME/anaconda

# Add Anaconda to current session's PATH
export PATH=$HOME/anaconda/bin:$PATH

conda install -c conda-forge -y imbalanced-learn
# git clone $1
# adding working directory to PYTHONPATH so each Python can locate src/,
# which will be used as a package for each approach
export PYTHONPATH=$PYTHONPATH:$(pwd)
