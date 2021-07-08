import time
from threading import Thread

import pytest

from ...base import DblEvent


@pytest.mark.unittest
class TestInteractionBaseThreading:
    # noinspection DuplicatedCode
    @pytest.mark.execution_timeout(5.0, method='thread')
    def test_dbl_event_open(self):
        event = DblEvent()
        assert event.is_close()
        assert not event.is_open()

        # Opening test
        _time_1, _time_2 = 0.0, 0.0

        def _run_1_wait_for_open():
            nonlocal _time_1
            event.wait_for_open()
            _time_1 = time.time()

        def _run_2_wait_for_open():
            nonlocal _time_2
            event.wait_for_open()
            _time_2 = time.time()

        _thread_1 = Thread(target=_run_1_wait_for_open)
        _thread_2 = Thread(target=_run_2_wait_for_open)

        _thread_1.start()
        _thread_2.start()

        time.sleep(0.2)
        assert event.is_close()
        assert not event.is_open()
        assert _time_1 == 0.0
        assert _time_2 == 0.0

        time.sleep(0.8)
        event.open()
        _thread_1.join()
        _thread_2.join()

        assert abs(time.time() - _time_1) < 0.3
        assert abs(time.time() - _time_2) < 0.3
        assert not event.is_close()
        assert event.is_open()

        # Closing test
        _time_1, _time_2 = 0.0, 0.0

        def _run_1_wait_for_close():
            nonlocal _time_1
            event.wait_for_close()
            _time_1 = time.time()

        def _run_2_wait_for_close():
            nonlocal _time_2
            event.wait_for_close()
            _time_2 = time.time()

        _thread_1 = Thread(target=_run_1_wait_for_close)
        _thread_2 = Thread(target=_run_2_wait_for_close)

        _thread_1.start()
        _thread_2.start()

        time.sleep(0.2)
        assert not event.is_close()
        assert event.is_open()
        assert _time_1 == 0.0
        assert _time_2 == 0.0

        time.sleep(0.8)
        event.close()
        _thread_1.join()
        _thread_2.join()

        assert abs(time.time() - _time_1) < 0.3
        assert abs(time.time() - _time_2) < 0.3
        assert event.is_close()
        assert not event.is_open()

    # noinspection DuplicatedCode
    @pytest.mark.execution_timeout(5.0, method='thread')
    def test_dbl_event_close(self):
        event = DblEvent(True)
        assert not event.is_close()
        assert event.is_open()

        # Closing test
        _time_1, _time_2 = 0.0, 0.0

        def _run_1_wait_for_close():
            nonlocal _time_1
            event.wait_for_close()
            _time_1 = time.time()

        def _run_2_wait_for_close():
            nonlocal _time_2
            event.wait_for_close()
            _time_2 = time.time()

        _thread_1 = Thread(target=_run_1_wait_for_close)
        _thread_2 = Thread(target=_run_2_wait_for_close)

        _thread_1.start()
        _thread_2.start()

        time.sleep(0.2)
        assert not event.is_close()
        assert event.is_open()
        assert _time_1 == 0.0
        assert _time_2 == 0.0

        time.sleep(0.8)
        event.close()
        _thread_1.join()
        _thread_2.join()

        assert abs(time.time() - _time_1) < 0.3
        assert abs(time.time() - _time_2) < 0.3
        assert event.is_close()
        assert not event.is_open()
