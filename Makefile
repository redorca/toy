POETRY_RELEASE := $$(sed -n -E "s/__version__ = '(.+)'/\1/p" poetry/__version__.py)

# lists all available targets
list:
	@sh -c "$(MAKE) -p no_targets__ | \
		awk -F':' '/^[a-zA-Z0-9][^\$$#\/\\t=]*:([^=]|$$)/ {\
			split(\$$1,A,/ /);for(i in A)print A[i]\
		}' | grep -v '__\$$' | grep -v 'make\[1\]' | grep -v 'Makefile' | sort"
# required for list
no_targets__:

run:
	@poetry install
	@poetry run python main.py $(ARGS)

clean:
	@rm -rf build dist .eggs *.egg-info
	@rm -rf .benchmarks .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -exec rm -rf {} +

format: clean
	@poetry run black poetry/ tests/

# test your application (tests in the tests/ directory)
test:
	@poetry install
	@poetry run pytest --cov=torque --cov-config .coveragerc tests/ -sq -rP $(ARGS)

release: build linux_release osx_release

update:
	@poetry update

install:
	@poetry update
	@poetry run pyinstaller -F openprotocol.spec --noconfirm

build:
	@poetry update
	@poetry run pyinstaller --name openprotocol main.py

proto:
	@protoc -I=protos --python_out=../OpenProtocol protos/torque/mids/*

docker_build:
	docker build -t toy:latest -f Dockerfile.deps .
	docker build -t toy_build:latest -f Dockerfile .

docker_run:
	docker run --rm -p 127.0.0.1:7496:7496 -v "${PWD}:/toy" -it toy_build

linux_release:
	docker run --rm --network host -v "${PWD}:/toy" -it toy_build
	install

osx_release:
	install

# run tests against all supported python versions
tox:
	@tox