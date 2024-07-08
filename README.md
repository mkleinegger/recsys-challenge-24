Add your Documantion here, the description can be found on [TUWEL](https://tuwel.tuwien.ac.at/mod/page/view.php?id=2281417)

# Recommender Systems Project of Group 33
To run the project after the hand-in, calling run.sh as described in the section "Hand In" should
suffice.
The provided wheel file can also alternatively be installed in a separate environment, the code
needed to to do that is also contained in the run.sh, but commented.

Execution permissions for run.sh are already set, you can redo them in case they are missing using:
```bash
chmod +x run.sh
```

## Setup & Installation
Environment management is done using poetry, deployment is done by installing a
pre-built wheel file.


### Development:
Make sure poetry and python are installed, the python version hast to match
the version specified in pyproject.toml.

Install the project using:
```shell
poetry install
```

### Deployment:
Build the wheel file using (for the hand in, the wheel should already be built):

```shell
poetry lock
poetry build
```

and install it on the target host using:

```shell
pip install dist/<name-of-wheel>.whl
```

The installation and running is already automated by the run.sh script.

### Hand In:
A wheel is already built, the installation is handled using [run.sh](./run.sh).

Create a virtual environment and load all dependencies:
```shell
python -m venv venv_group_33
source venv_group_33/bin/activate
pip install dist/group_33-*.whl
```

Then call run.sh to execute dkn:
```shell
./run.sh dkn
```

For the second model, lstur:
```shell
./run.sh lstur
```

If the predictions should not be done using the pretrained model
but rather be retrained, call the script in the following way:
```shell
./run.sh dkn -t
```
This sets the TRAIN environment variable, which triggers the training phase inside the
executed notebook.
Equivalently this can also be done for lstur:
```shell
./run.sh lstur -t
```

## Predictions and Models

The pretrained models are saved in the [submission/models](~/shared/194.035-2024S/groups/Gruppe_33/Group_33/submission/models) folder of the shared directory. Each model has its separate sub directory, which contains the weights and all other information required to load the model.

Predictions of each model are contained in the [submission/predictions](~/shared/194.035-2024S/groups/Gruppe_33/Group_33/submission/predictions) directory in the format expected by the challenge. 
