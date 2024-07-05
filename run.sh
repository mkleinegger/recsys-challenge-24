#!/bin/sh
#
# Install dependencies and predict using one of the
# 2 implemented models (besides baseline): lstur and dkn
# (where dkn achieved best performances).
#
# Call this script either with dkn or with lstur
# to run the specified model. By default dkn is run.
# To switch to train mode, append -t at the end. If -t
# is not specified, the pretrained model is used.
# example: ./run.sh lstur -t

# venv creation:
# python -m venv venv_group_33
# activate it
# ./venv_group_33/bin/activate


# install dependencies & package
pip install dist/group_33-*.whl

# now, either execute entrypoint:
# python -m group_33.main

# or use entrypoint directly:
# predict_dkn

# Setup environment variables:
# if TRAIN is set, the model is trained before prediction,
# else the pretrained model is loaded and used for predictions.


# set model
if [ "$1" = "lstur" ]; then
    NB_PATH="./notebooks/lstur.ipynb"
else
    NB_PATH="./notebooks/dkn.ipynb"
fi

# set train mode
if [ "$2" = "-t" ]; then
    export TRAIN="True"
else
    export TRAIN=
fi

echo "running notebook at location: $NB_PATH with TRAIN=$TRAIN."
# or execute notebook directly:
# nbconvert --execute /path/to/notebook
#
# or:
jupyter execute --inplace $NB_PATH
echo "Checkout the notebook at $NB_PATH to see more detailed results of the run."
