# Go to home directory
# cd ~

# You can change what anaconda version you want at 
# https://repo.continuum.io/archive/
# wget https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
# bash Anaconda3-4.2.0-Linux-x86_64.sh -b -p ~/anaconda
# rm Anaconda3-4.2.0-Linux-x86_64.sh
echo 'export PATH="~/anaconda/bin:$PATH"' >> ~/.bashhrc 

export PATH=~/anaconda/bin:$PATH

# Reload default profile
# source ~/.bashrc

. ~/.zshrc

conda update conda

# Install pytorch cuda package 
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
