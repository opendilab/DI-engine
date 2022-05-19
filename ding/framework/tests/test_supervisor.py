from time import sleep
from typing import List
import pytest
from ding.framework.supervisor import RecvPayload, SendPayload, Supervisor, ChildType


class MockEnv():

    def __init__(self, _) -> None:
        self._counter = 0

    def step(self, _):
        self._counter += 1
        return self._counter

    @property
    def action_space(self):
        return 3

    def block(self):
        sleep(10)


@pytest.mark.unittest
@pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
def test_supervisor(type_):
    sv = Supervisor(type_=type_)
    for _ in range(3):
        sv.register(MockEnv, "AnyArgs")
    sv.start_link()

    for env_id in range(len(sv._children)):
        sv.send(SendPayload(proc_id=env_id, method="step", args=["any action"]))

    recv_states: List[RecvPayload] = []
    for _ in range(3):
        recv_states.append(sv.recv())

    assert sum([payload.proc_id for payload in recv_states]) == 3
    assert all([payload.data == 1 for payload in recv_states])

    # Test recv_all
    req_ids = []
    for env_id in range(len(sv._children)):
        payload = SendPayload(
            proc_id=env_id,
            method="step",
            args=["any action"],
        )
        req_ids.append(payload.req_id)
        sv.send(payload)

    ## Only wait for last two messages, keep the first one in the queue.
    recv_payloads = sv.recv_all(req_ids=req_ids[1:])
    assert len(recv_payloads) == 2
    for req_id, payload in zip(req_ids[1:], recv_payloads):
        assert req_id == payload.req_id

    recv_payload = sv.recv()
    assert recv_payload.req_id == req_ids[0]

    assert len(sv.action_space) == 3
    assert all(a == 3 for a in sv.action_space)

    sv.shutdown()


class MockCrashEnv(MockEnv):

    def step(self, _):
        super().step(_)
        if self._counter == 2:
            raise Exception("Ohh")

        return self._counter


@pytest.mark.unittest()
@pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
def test_crash_supervisor(type_):
    sv = Supervisor(type_=type_)
    for _ in range(2):
        sv.register(MockEnv, "AnyArgs")
    sv.register(MockCrashEnv, "AnyArgs")
    sv.start_link()

    # Send 6 messages, will cause the third subprocess crash
    for env_id in range(len(sv._children)):
        for _ in range(2):
            sv.send(SendPayload(proc_id=env_id, method="step", args=["any action"]))

    # Find the error mesasge
    recv_states: List[RecvPayload] = []
    for _ in range(6):
        recv_payload = sv.recv(ignore_err=True)
        if recv_payload.err:
            sv._children[recv_payload.proc_id].restart()
        recv_states.append(recv_payload)
    assert any([isinstance(payload.err, Exception) for payload in recv_states])

    # Resume
    for env_id in range(len(sv._children)):
        sv.send(SendPayload(proc_id=env_id, method="step", args=["any action"]))
    recv_states: List[RecvPayload] = []
    for _ in range(3):
        recv_states.append(sv.recv())

    # 3 + 3 + 1
    assert sum([p.data for p in recv_states]) == 7

    with pytest.raises(Exception):
        sv.send(SendPayload(proc_id=2, method="step", args=["any action"]))
        sv.recv(ignore_err=False)

    sv.shutdown()


@pytest.mark.unittest
@pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
def test_recv_all(type_):
    sv = Supervisor(type_=type_)
    for _ in range(3):
        sv.register(MockEnv, "AnyArgs")
    sv.start_link()

    # Test recv_all
    req_ids = []
    for env_id in range(len(sv._children)):
        payload = SendPayload(
            proc_id=env_id,
            method="step",
            args=["any action"],
        )
        req_ids.append(payload.req_id)
        sv.send(payload)

    retry_times = {env_id: 0 for env_id in range(len(sv._children))}

    def recv_callback(recv_payload: RecvPayload, req_ids: List[str]):
        if retry_times[recv_payload.proc_id] == 2:
            return
        retry_times[recv_payload.proc_id] += 1
        payload = SendPayload(proc_id=recv_payload.proc_id, method="step", args={"action"})
        sv.send(payload)
        req_ids.append(payload.req_id)

    recv_payloads = sv.recv_all(req_ids=req_ids, callback=recv_callback)
    assert len(recv_payloads) == 3
    assert all([v == 2 for v in retry_times.values()])

    sv.shutdown()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("type_", [ChildType.PROCESS, ChildType.THREAD])
def test_timeout(type_):
    sv = Supervisor(type_=type_)
    for _ in range(3):
        sv.register(MockEnv, "AnyArgs")
    sv.start_link()

    req_ids = []
    for env_id in range(len(sv._children)):
        payload = SendPayload(proc_id=env_id, method="block")
        req_ids.append(payload.req_id)
        sv.send(payload)

    # Test timeout exception
    with pytest.raises(TimeoutError):
        sv.recv_all(req_ids=req_ids, timeout=1)
    sv.shutdown(timeout=1)

    # Test timeout with ignore error
    sv.start_link()
    req_ids = []

    ## 0 is block
    payload = SendPayload(proc_id=0, method="block")
    req_ids.append(payload.req_id)
    sv.send(payload)

    ## 1 is step
    payload = SendPayload(proc_id=1, method="step", args=[""])
    req_ids.append(payload.req_id)
    sv.send(payload)

    # Will receiving a payload and an timeout error object
    payloads = sv.recv_all(req_ids=req_ids, timeout=1, ignore_err=True)
    assert isinstance(payloads[0], TimeoutError)
    assert isinstance(payloads[1], RecvPayload)

    sv.shutdown(timeout=1)
