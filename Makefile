PACKAGE=nlpiper
UNIT_TESTS=tests

all: static-tests coverage

.PHONY: all

style:
		###### Running style analysis ######
		poetry run flake8 $(PACKAGE) $(UNIT_TESTS)

typecheck:
		###### Running static type analysis ######
		poetry run mypy $(PACKAGE)

doccheck:
		###### Running documentation analysis ######
		poetry run pydocstyle -v $(PACKAGE)

static-tests: style typecheck doccheck

unit-tests:
		###### Running unit tests ######
		poetry run pytest -v $(UNIT_TESTS)

coverage:
		###### Running coverage analysis with JUnit xml export ######
		poetry run pytest -v --cov-report term-missing --cov $(PACKAGE)

coverage-html:
		###### Running coverage analysis with html export ######
		poetry run pytest -v --cov-report html --cov $(PACKAGE)
		open htmlcov/index.html
