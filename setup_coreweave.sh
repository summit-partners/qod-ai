#!/usr/bin/env bash

# Setup and load asdf
if [ ! -d "$HOME/.asdf" ]; then
  git clone https://github.com/asdf-vm/asdf.git "$HOME/.asdf" --branch v0.12.0
  echo '. "$HOME/.asdf/asdf.sh"' >>~/.bashrc
  echo '. "$HOME/.asdf/completions/asdf.bash"' >>~/.bashrc
fi

# https://stackoverflow.com/questions/75080993/dbuserrorresponse-while-running-poetry-install
echo 'export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring' >>~/.bashrc
source "$HOME/.bashrc"

# Install dev dependencies (for installing Python)
sudo apt update
sudo apt install -y build-essential checkinstall
sudo apt install -y libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev liblzma-dev libffi-dev

make setup
echo 'Run \`source \"$HOME/.bashrc\"\` to refresh the environment'
