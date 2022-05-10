CI ?=

# Directory variables
DING_DIR   ?= ./ding
DIZOO_DIR  ?= ./dizoo
RANGE_DIR  ?=
TEST_DIR   ?= $(if ${RANGE_DIR},${RANGE_DIR},${DING_DIR})
COV_DIR    ?= $(if ${RANGE_DIR},${RANGE_DIR},${DING_DIR})
FORMAT_DIR ?= $(if ${RANGE_DIR},${RANGE_DIR},${DING_DIR})
PLATFORM_TEST_DIR   ?= $(if ${RANGE_DIR},${RANGE_DIR},${DING_DIR}/entry/tests/test_serial_entry.py ${DING_DIR}/entry/tests/test_serial_entry_onpolicy.py)

# Workers command
WORKERS         ?= 2
WORKERS_COMMAND := $(if ${WORKERS},-n ${WORKERS} --dist=loadscope,)

# Duration command
DURATIONS         ?= 10
DURATIONS_COMMAND := $(if ${DURATIONS},--durations=${DURATIONS},)

# Rerun command
RERUN       ?=
RERUN_DELAY ?=

CI_DEFAULT_RERUN          := 5
LOCAL_DEFAULT_RERUN       := 3
DEFAULT_RERUN             ?= $(if ${CI},${CI_DEFAULT_RERUN},${LOCAL_DEFAULT_RERUN})
ACTUAL_RERUN              := $(if ${RERUN},${RERUN},${DEFAULT_RERUN})

CI_DEFAULT_RERUN_DELAY    := 10
LOCAL_DEFAULT_RERUN_DELAY := 5
DEFAULT_RERUN_DELAY       ?= $(if ${CI},${CI_DEFAULT_RERUN_DELAY},${LOCAL_DEFAULT_RERUN_DELAY})
ACTUAL_RERUN_DELAY        := $(if ${RERUN_DELAY},${RERUN_DELAY},${DEFAULT_RERUN_DELAY})

RERUN_COMMAND := $(if ${CI}${ACTUAL_RERUN},--reruns ${ACTUAL_RERUN} --reruns-delay ${ACTUAL_RERUN_DELAY},)

docs:
	$(MAKE) -C ${DING_DIR}/docs html

unittest:
	pytest ${TEST_DIR} \
		--cov-report=xml \
		--cov-report term-missing \
		--cov=${COV_DIR} \
		${DURATIONS_COMMAND} \
		${RERUN_COMMAND} \
		${WORKERS_COMMAND} \
		-sv -m unittest \

algotest:
	pytest ${TEST_DIR} \
		${DURATIONS_COMMAND} \
		-sv -m algotest

cudatest:
	pytest ${TEST_DIR} \
		-sv -m cudatest

envpooltest:
	pytest ${TEST_DIR} \
		-sv -m envpooltest

dockertest:
	${DING_DIR}/scripts/docker-test-entry.sh

platformtest:
	pytest ${PLATFORM_TEST_DIR} \
		--cov-report term-missing \
		--cov=${COV_DIR} \
		${WORKERS_COMMAND} \
		-sv -m unittest \

benchmark:
	pytest ${TEST_DIR} \
		--durations=0 \
		-sv -m benchmark

test: unittest  # just for compatibility, can be changed later

cpu_test: unittest algotest benchmark

all_test: unittest algotest cudatest benchmark

format:
	yapf --in-place --recursive -p --verbose --style .style.yapf ${FORMAT_DIR}
format_test:
	bash format.sh ${FORMAT_DIR} --test
flake_check:
	flake8 ${FORMAT_DIR}
