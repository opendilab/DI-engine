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


@pytest.mark.unittest
def test_supervisor():

    def _test_supervisor(type_):
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

    _test_supervisor(ChildType.PROCESS)
    _test_supervisor(ChildType.THREAD)


class MockCrashEnv(MockEnv):

    def step(self, _):
        super().step(_)
        if self._counter == 2:
            raise Exception("Ohh")

        return self._counter


@pytest.mark.unittest()
def test_crash_supervisor():

    def _test_crash_supervisor(type_):
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
            recv_states.append(sv.recv(ignore_err=True))
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

    _test_crash_supervisor(ChildType.PROCESS)
    _test_crash_supervisor(ChildType.THREAD)
