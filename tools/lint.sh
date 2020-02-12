#!/bin/sh

if ! command -v isort &> /dev/null
then
    echo "isort not found."
    exit 1
fi

if ! command -v flake8 &> /dev/null
then
    echo "flake8 not found."
    exit 1
fi

for file in $(find . -name "*.py")
do
    isort $file
    flake8 $file
done
