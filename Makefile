## Installs dependencies from Pipefile and activate virtual environment
setup-env:
	pip install pipenv 
	pipenv install 
	pipenv shell


test:
	pytest

