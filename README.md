# Jeremy Mann's Machine Learning Tools


Random collection of ML tools (transformers, etc) primarily intended for personal use.

More complete documentation may be found [here](https://jmann277.github.io/jers_ml_tools).

# Usage

In order to ensure reproducibility, we have chosen to use [Pipenv](https://pipenv-fork.readthedocs.io/en/latest/basics.html) to manage our dependencies. These dependencies may be found inside of `Pipfile`.

You can install Pipenv with Homebrew using the following command:

```bash
$ brew intall pipenv
```

You can install the dependencies and create a virtual environment by navigating to the `jers_ml_tools` folder and executing the following command:

```bash
$ pipenv install
```

This will install all the project's dependencies (e.g. 'numpy >= 1.18.1') and create a virtual environment. You can activate the environment using the following command:

```bash
$ pipenv shell
```

and can deactivate the environment using the following command: 

```bash
$ exit
```


