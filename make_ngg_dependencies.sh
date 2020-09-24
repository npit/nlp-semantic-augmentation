#/usr/bin/env bash

# workaround if pygraphviz errors out
# # install graphviz
# sudo apt install graphviz-dev
# 
# headers="/usr/incluce/graphviz"
# libs="/usr/lib/graphviz"
# 
# # install pip package
# python3 -m pip install pygraphviz --install-option="--include-path=${headers}" --install-option="--library-path=${libs}"

# install pyinsect
python3 -m pip  install --user -i https://test.pypi.org/simple/ pyinsect-ggianna==0.0.37
