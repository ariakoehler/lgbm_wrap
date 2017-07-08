#!/bin/bash

targets=('classification' 'regression')

echo "--------------------------------------------------"
printf "Setting up extensions to autosklearn\n\n\n"
base=
for name in ${targets[@]}
do
    printf "Installing and adding $name component\n\n\n"
    dest="../autosklearn/pipeline/components/$name/"
    cp ./$name/{lightgb_machine.py,add_script.py} $dest
    pushd $dest
    if hash python3; then
	python3 ./add_script.py
    else
	python ./add_script.py
    fi
    rm add_script.py
    popd
done	    
