#!/bin/bash
# Run from root with `. scripts/activate.sh`

# Try to deactivate conda, but don't fail if it's not available
command -v conda >/dev/null 2>&1 && conda deactivate

# .venv lives in the parent's parent directory
source $(dirname $(dirname "$BASH_SOURCE"))/.venv/bin/activate
