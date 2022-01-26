#!bin/bash
set -ev

export PYTHONPATH=$PYTHONPATH:.

echo $(pwd)

if [ $TRAVIS_EVENT_TYPE = "cron" ]
then
	cd $TRAVIS_BUILD_DIR/test/regression
	python check_tutorials.py

	cd $TRAVIS_BUILD_DIR
  	docker run -u root docker.io/pmgbergen/porepylib:py27 /bin/sh -c "cd /home/porepy/porepy-src; python setup.py test;sh ./test/regression/run_cron_test.sh"
  	 docker run -u root docker.io/pmgbergen/porepylib:py36 /bin/sh -c "cd /home/porepy/porepy-src; python setup.py test;sh ./test/regression/run_cron_test.sh"
fi
