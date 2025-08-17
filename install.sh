# Unset proxy variables and Conda proxy settings
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
/opt/anaconda3/bin/conda config --remove-key proxy_servers

# Check for existing Conda installation
if ! command -v conda >/dev/null 2>&1; then
    mkdir -p ~/miniconda3
    curl -o ~/miniconda3/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init zsh
else
    CONDA_PATH=$(conda info --base)
    $CONDA_PATH/bin/conda init zsh
fi
source ~/.zshrc
$CONDA_PATH/bin/conda create -n arena-env python=3.11 -y
source $CONDA_PATH/etc/profile.d/conda.sh
conda activate arena-env
$CONDA_PATH/envs/arena-env/bin/pip install -r /Users/admin/Documents/GitHub/ARENA_3.0/requirements.txt 
$CONDA_PATH/bin/conda install -n arena-env ipykernel --update-deps --force-reinstall -y