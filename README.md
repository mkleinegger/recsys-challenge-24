Add your Documantion here the description can be found on [TUWEL](https://tuwel.tuwien.ac.at/mod/page/view.php?id=2281417)

# Recommender Systems Project of Group 33

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

The installation and runnign is already automated by the run.sh script.
