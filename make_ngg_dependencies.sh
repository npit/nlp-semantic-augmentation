#/usr/bin/env bash

# install graphviz
sudo apt install graphviz

headers="/usr/invluce/graphviz"
libs="/usr/lib/graphviz"

# install pip package
python3 -m pip install pygraphviz --install-option="--include-path=${headers}" --install-option="--library-path=${libs}"
