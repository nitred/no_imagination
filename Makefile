manual_recreate_env:
	conda env create --force -f dev_environment.yml

PROJECT_NAME = "no_imagination"
ANACONDA_VERSION = "3"

ROOT_DIR = $(shell echo ~/anaconda$(ANACONDA_VERSION))
ROOT_PYTHON= "$(ROOT_DIR)/bin/python"
ROOT_CONDA = "$(ROOT_DIR)/bin/conda"
ENV_DIR = "$(ROOT_DIR)/envs/$(PROJECT_NAME)"
ENV_BIN = "$(ENV_DIR)/bin"
ENV_PIP = "$(ENV_BIN)/pip"
ENV_PYTHON = "$(ENV_BIN)/python"

recreate_env:
	$(ROOT_CONDA) env create --force -f dev_environment.yml && \
	$(ENV_PIP) uninstall -y $(PROJECT_NAME) || true && \
	$(ENV_PIP) install -e . --upgrade

test_env_python:
	@echo $(shell "$(ENV_PYTHON)" --version)
