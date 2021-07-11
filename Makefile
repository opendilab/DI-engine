CI := $(shell echo ${CI})

WORKERS         ?= 4
WORKERS_COMMAND := $(if ${WORKERS},-n ${WORKERS} --dist=loadscope,)

DURATIONS         ?= 10
DURATIONS_COMMAND := $(if ${DURATIONS},--durations=${DURATIONS},)

RANGE_DIR  ?=
TEST_DIR   ?= $(if ${RANGE_DIR},${RANGE_DIR},./ding/rl_utils ./ding/interaction ./ding/utils ./ding/torch_utils ./ding/model ./ding/league ./ding/envs)
COV_DIR    ?= $(if ${RANGE_DIR},${RANGE_DIR},./ding)
FORMAT_DIR ?= $(if ${RANGE_DIR},${RANGE_DIR},./ding)
PLATFORM_TEST_DIR   ?= $(if ${RANGE_DIR},${RANGE_DIR},./ding/entry/tests/test_serial_entry.py ./ding/entry/tests/test_serial_entry_reward_model.py ./ding/entry/tests/test_serial_entry_il.py)

docs:
	$(MAKE) -C ./ding/docs html

unittest:
	pytest ${TEST_DIR} \
		--cov-report term-missing \
		--cov=${COV_DIR} \
		${WORKERS_COMMAND} \
		-sv -m unittest \
	    --execution-timeout=600

algotest:
	pytest ${TEST_DIR} \
		${DURATIONS_COMMAND} \
		-sv -m algotest

cudatest:
	pytest ${TEST_DIR} \
		-sv -m cudatest

platformtest:
	pytest ${PLATFORM_TEST_DIR} \
		${WORKERS_COMMAND} \
		-sv -m unittest \
	    --execution-timeout=900

benchmark:
	pytest ${./ding} \
		--durations=0 \
		-sv -m benchmark

test: unittest  # just for compatibility, can be changed later

cpu_test: unittest algotest benchmark

all_test: unittest algotest cudatest benchmark

format:
	yapf --in-place --recursive -p --verbose --style .style.yapf ${FORMAT_DIR}
format_test:
	bash format.sh ./ding --test
flake_check:
	flake8 ./ding
