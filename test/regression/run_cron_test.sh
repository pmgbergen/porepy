#!bin/bash
set -ev

export PYTHONPATH=$PYTHONPATH:.

echo $(pwd)

if [ $TRAVIS_EVENT_TYPE = "cron" ]
then
	cd $TRAVIS_BUILD_DIR/test/regression
	python check_tutorials.py
fi
