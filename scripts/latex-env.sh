#!/usr/bin/env bash

# Activate the project-local TeX Live installation in the current shell.
# Usage: source scripts/latex-env.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Please source this script: source scripts/latex-env.sh" >&2
    exit 1
fi

_latex_env_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_latex_env_project_root="$(cd -- "${_latex_env_script_dir}/.." && pwd)"

export TEXLIVE_ROOT="${_latex_env_project_root}/.local/texlive/2026"
export PATH="${TEXLIVE_ROOT}/bin/x86_64-linux:${PATH}"
export MANPATH="${TEXLIVE_ROOT}/texmf-dist/doc/man${MANPATH:+:${MANPATH}}"
export INFOPATH="${TEXLIVE_ROOT}/texmf-dist/doc/info${INFOPATH:+:${INFOPATH}}"

unset _latex_env_script_dir _latex_env_project_root
