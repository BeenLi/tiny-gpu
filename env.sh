#!/usr/bin/env bash
# Source this before running the tiny-gpu simulations:
#   source env.sh && make test_matadd
# It activates the Python 3.11 venv that has cocotb 1.9.2 (the global
# ~/.local cocotb is 2.0 on python3.7 and is NOT compatible with this repo).
export PATH="$HOME/.local/bin:$PATH"
# shellcheck disable=SC1090
source "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/.venv/bin/activate"
