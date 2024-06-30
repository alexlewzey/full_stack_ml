#!/bin/bash
echo 'source $(pwd)/.venv/bin/activate' >> $HOME/.bashrc
pre-commit install
poetry install