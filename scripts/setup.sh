#!/bin/bash
# Run from root with `. scripts/setup.sh`

uv sync
source .venv/bin/activate
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=bbhamux
