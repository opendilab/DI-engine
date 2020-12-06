CI := $(shell echo ${CI})

CI_DEFAULT_WORKERS := 8
LOCAL_DEFAULT_WORKERS :=
WORKERS ?= $(if ${CI},${CI_DEFAULT_WORKERS},${LOCAL_DEFAULT_WORKERS})
WORKERS_COMMAND := $(if ${WORKERS},-n ${WORKERS},)

TEST_DIR ?= ./nervex ./app_zoo

info:
	echo ${TEST_DIR}

docs:
	$(MAKE) -C ./nervex/docs html

test:
	pytest ${TEST_DIR} \
		--cov-report term-missing --cov=./nervex \
		${WORKERS_COMMAND} -sv -m unittest

format:
	bash format.sh
format_test:
	bash format.sh --test
flake_check:
	flake8 ./nervex
