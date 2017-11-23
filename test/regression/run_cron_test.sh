#!bin/bash
set -ev

export PYTHONPATH=$PYTHONPATH:.
if [ $TRAVIS_EVENT_TYPE = "cron" ]
then
	python check_tutorials.py
fi
