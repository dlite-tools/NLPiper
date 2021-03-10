PACKAGE=nlpiper
UNIT_TESTS=tests

all: static-tests coverage

.PHONY: all

style:
		###### Running style analysis ######
		pipenv run flake8 $(PACKAGE)

typecheck:
		###### Running static type analysis ######
		pipenv run mypy $(PACKAGE)

doccheck:
		###### Running documentation analysis ######
		pipenv run pydocstyle -v $(PACKAGE)

static-tests: style typecheck doccheck

unit-tests:
		###### Running unit tests ######
		pipenv run pytest -v $(UNIT_TESTS)

coverage:
		###### Running coverage analysis with JUnit xml export ######
		pipenv run pytest -v --cov-report term-missing --cov $(PACKAGE)

coverage-html:
		###### Running coverage analysis with html export ######
		pipenv run pytest -v --cov-report html --cov $(PACKAGE)
		open htmlcov/index.html

publish:
		###### Publish package to Pypi server on Nexus ######
		pipenv run python setup.py sdist bdist_wheel
		pipenv run twine upload --non-interactive -u ${PYPI_USER} -p ${PYPI_PASS} --repository-url ${PYPI_URL} dist/*
