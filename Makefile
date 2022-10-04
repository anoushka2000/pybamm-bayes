.PHONY: install
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt -U
	pip install -e . --no-deps

.PHONY: black
black:
	black src --line-length=120
	black tests --line-length=120

.PHONY: test_black
test_black:
	black src --line-length=120 --check
	black tests --line-length=120 --check

.PHONY: isort
isort:
	isort src --profile black --line-length=120
	isort tests --profile black --line-length=120

.PHONY: test_isort
test_isort:
	isort src -c --profile black --line-length=120
	isort tests -c --profile black --line-length=120

.PHONY: flake8
flake8:
	flake8 src --count --show-source --statistics

.PHONY: unittest
test:
	python -m unittest

.PHONY: format
format:
	make black
	make isort
	make flake8

.PHONY: test
test:
	make install
	make isort
	make black
	make flake8
	make unittest