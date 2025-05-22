#!/bin/bash
# Try to deactivate conda, but don't fail if it's not available
# run with `. make/activate.sh`

command -v conda >/dev/null 2>&1 && conda deactivate
source .venv/bin/activate
