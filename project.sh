#!/usr/bin/env bash

if [ $1 == "clean" ]; then
    rm -rf build dist deepgen.egg-info
    echo "This project has been cleaned!!!"
elif [ $1 == "build" ]; then
    python3 setup.py sdist bdist_wheel
    twine check dist/*
    echo "This project was built!!!"
elif [ $1 == "release" ]; then
    python3 setup.py sdist bdist_wheel
    twine check dist/*
    twine upload dist/*
    echo "This project has been released!!!"
elif [ $1 == "bumpversion" ]; then
    bumpversion --current-version $2 $3 setup.py deepgen/__init__.py --allow-dirty
    echo "Updated version!!!"
else
    echo "Unknow argument!!!"
    echo $@
fi