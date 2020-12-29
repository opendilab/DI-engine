CI := $(shell echo ${CI})

CI_DEFAULT_WORKERS := 8
LOCAL_DEFAULT_WORKERS :=
WORKERS ?= $(if ${CI},${CI_DEFAULT_WORKERS},${LOCAL_DEFAULT_WORKERS})
WORKERS_COMMAND := $(if ${WORKERS},-n ${WORKERS},)

RANGE_DIR ?=
TEST_DIR  ?= $(if ${RANGE_DIR},${RANGE_DIR},./nervex)
COV_DIR   ?= $(if ${RANGE_DIR},${RANGE_DIR},./nervex)

info:
	echo ${TEST_DIR}

docs:
	$(MAKE) -C ./nervex/docs html

unittest:
	pytest ${TEST_DIR} \
		--cov-report term-missing --cov=${COV_DIR} \
		${WORKERS_COMMAND} -sv -m unittest

algotest:
	pytest ${TEST_DIR} \
		--durations=10 \
		${WORKERS_COMMAND} -sv -m algotest

cudatest:  # do not use this yet, TODO: complete this part
	echo "this is empty cuda test"

benchmark:
	pytest ${TEST_DIR} \
		--durations=0 \
		-sv -m benchmark

test: unittest  # just for compatibility, can be changed later

cpu_test: unittest algotest benchmark

all_test: unittest algotest cudatest benchmark

format:
	bash format.sh
format_test:
	bash format.sh --test
flake_check:
	flake8 ./nervex
