python=python3

build:
	$(python) setup.py build

upload:
	$(python) setup.py sdist bdist_wheel
	twine upload dist/*

clean:
	@rm -rf build dist nnieqat.egg-info

test:
	$(python) /usr/bin/nosetests -s tests --nologcapture

lint:
	pylint nnieqat --reports=n

lintfull:
	pylint nnieqat

install:
	$(python) setup.py install 

uninstall:
	$(python) setup.py install --record install.log
	cat install.log | xargs rm -rf 
	@rm install.log

