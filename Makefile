CI := $(shell echo ${CI})

CI_DEFAULT_WORKERS := 8
LOCAL_DEFAULT_WORKERS :=
WORKERS ?= $(if ${CI},${CI_DEFAULT_WORKERS},${LOCAL_DEFAULT_WORKERS})
WORKERS_COMMAND := $(if ${WORKERS},-n ${WORKERS},)

RANGE_DIR ?=
TEST_DIR  ?= $(if ${RANGE_DIR},${RANGE_DIR},./nervex ./app_zoo)
COV_DIR   ?= $(if ${RANGE_DIR},${RANGE_DIR},./nervex)

info:
	echo ${TEST_DIR}

docs:
	$(MAKE) -C ./nervex/docs html

test:
	pytest ${TEST_DIR} \
		--cov-report term-missing --cov=${COV_DIR} \
		${WORKERS_COMMAND} -sv -m unittest

format:
	bash format.sh
format_test:
	bash format.sh --test
flake_check:
	flake8 ./nervex
