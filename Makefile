CI := $(shell echo ${CI})

WORKERS         ?=
WORKERS_COMMAND := $(if ${WORKERS},-n ${WORKERS},)

DURATIONS         ?= 10
DURATIONS_COMMAND := $(if ${DURATIONS},--duration=${DURATIONS},)

RANGE_DIR ?=
TEST_DIR  ?= $(if ${RANGE_DIR},${RANGE_DIR},./nervex)
COV_DIR   ?= $(if ${RANGE_DIR},${RANGE_DIR},./nervex)

docs:
	$(MAKE) -C ./nervex/docs html

unittest:
	pytest ${TEST_DIR} \
		--cov-report term-missing \
		--cov=${COV_DIR} \
		${WORKERS_COMMAND} \
		-sv -m unittest

algotest:
	pytest ${TEST_DIR} \
		${DURATIONS_COMMAND} \
		${WORKERS_COMMAND} \
		-sv -m algotest

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
