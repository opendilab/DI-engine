# Copyright (c) 2020 DeNA Co., Ltd.
# Licensed under The MIT License [see LICENSE for details]

import io
import time
import struct
import socket
import pickle
import threading
import queue
import select
import multiprocessing as mp


def send_recv(conn, sdata):
    conn.send(sdata)
    rdata = conn.recv()
    return rdata


class PickledConnection:
    def __init__(self, conn):
        self.conn = conn

    def __del__(self):
        self.close()

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def fileno(self):
        return self.conn.fileno()

    def _recv(self, size):
        buf = io.BytesIO()
        while size > 0:
            chunk = self.conn.recv(size)
            if len(chunk) == 0:
                raise ConnectionResetError
            size -= len(chunk)
            buf.write(chunk)
        return buf

    def recv(self):
        buf = self._recv(4)
        size, = struct.unpack("!i", buf.getvalue())
        buf = self._recv(size)
        return pickle.loads(buf.getvalue())

    def _send(self, buf):
        size = len(buf)
        while size > 0:
            n = self.conn.send(buf)
            size -= n
            buf = buf[n:]

    def send(self, msg):
        buf = pickle.dumps(msg)
        n = len(buf)
        header = struct.pack("!i", n)
        if n > 16384:
            chunks = [header, buf]
        elif n > 0:
            chunks = [header + buf]
        else:
            chunks = [header]
        for chunk in chunks:
            self._send(chunk)


def open_socket_connection(port, reuse=False):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(
        socket.SOL_SOCKET, socket.SO_REUSEADDR,
        sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) | 1
    )
    sock.bind(('', int(port)))
    return sock


def accept_socket_connection(sock):
    try:
        conn, _ = sock.accept()
        return PickledConnection(conn)
    except socket.timeout:
        return None


def listen_socket_connections(n, port):
    sock = open_socket_connection(port)
    sock.listen(n)
    return [accept_socket_connection(sock) for _ in range(n)]


def connect_socket_connection(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, int(port)))
    except ConnectionRefusedError:
        print('failed to connect %s %d' % (host, port))
    return PickledConnection(sock)


def accept_socket_connections(port, timeout=None, maxsize=1024):
    sock = open_socket_connection(port)
    sock.listen(maxsize)
    sock.settimeout(timeout)
    cnt = 0
    while cnt < maxsize:
        conn = accept_socket_connection(sock)
        if conn is not None:
            cnt += 1
        yield conn


def open_multiprocessing_connections(num_process, target, args_func):
    # open connections
    s_conns, g_conns = [], []
    for _ in range(num_process):
        conn0, conn1 = mp.connection.Pipe(duplex=True)
        s_conns.append(conn0)
        g_conns.append(conn1)

    # open workers
    for i, conn in enumerate(g_conns):
        mp.Process(target=target, args=args_func(i, conn)).start()
        conn.close()

    return s_conns


class MultiProcessWorkers:
    def __init__(self, func, send_generator, num, postprocess=None, buffer_length=512, num_receivers=1):
        self.send_generator = send_generator
        self.postprocess = postprocess
        self.buffer_length = buffer_length
        self.num_receivers = num_receivers
        self.conns = []
        self.send_cnt = {}
        self.shutdown_flag = False
        self.lock = threading.Lock()
        self.output_queue = queue.Queue(maxsize=8)
        self.threads = []

        for i in range(num):
            conn0, conn1 = mp.Pipe(duplex=True)
            mp.Process(target=func, args=(conn1, i)).start()
            conn1.close()
            self.conns.append(conn0)
            self.send_cnt[conn0] = 0

    def shutdown(self):
        self.shutdown_flag = True
        for thread in self.threads:
            thread.join()

    def recv(self):
        return self.output_queue.get()

    def start(self):
        self.threads.append(threading.Thread(target=self._sender))
        for i in range(self.num_receivers):
            self.threads.append(threading.Thread(target=self._receiver, args=(i,)))
        for thread in self.threads:
            thread.start()

    def _sender(self):
        print('start sender')
        while not self.shutdown_flag:
            total_send_cnt = 0
            for conn, cnt in self.send_cnt.items():
                if cnt < self.buffer_length:
                    conn.send(next(self.send_generator))
                    self.lock.acquire()
                    self.send_cnt[conn] += 1
                    self.lock.release()
                    total_send_cnt += 1
            if total_send_cnt == 0:
                time.sleep(0.01)
        print('finished sender')

    def _receiver(self, index):
        print('start receiver %d' % index)
        conns = [conn for i, conn in enumerate(self.conns) if i % self.num_receivers == index]
        while not self.shutdown_flag:
            tmp_conns = mp.connection.wait(conns)
            for conn in tmp_conns:
                data, cnt = conn.recv()
                if self.postprocess is not None:
                    data = self.postprocess(data)
                while not self.shutdown_flag:
                    try:
                        self.output_queue.put(data, timeout=0.3)
                        self.lock.acquire()
                        self.send_cnt[conn] -= cnt
                        self.lock.release()
                        break
                    except queue.Full:
                        pass
        print('finished receiver %d' % index)


class QueueCommunicator:
    def __init__(self, conns=[]):
        self.input_queue = queue.Queue(maxsize=256)
        self.output_queue = queue.Queue(maxsize=256)
        self.conns, self.conn_ids = {}, 0
        for conn in conns:
            self.add(conn)
        self.shutdown_flag = False
        self.threads = [
            threading.Thread(target=self._send_thread),
            threading.Thread(target=self._recv_thread),
        ]
        for thread in self.threads:
            thread.start()

    def shutdown(self):
        self.shutdown_flag = True
        for thread in self.threads:
            thread.join()

    def recv(self):
        return self.input_queue.get()

    def send(self, conn, send_data):
        self.output_queue.put((conn, send_data))

    def add(self, conn):
        self.conns[conn] = self.conn_ids
        self.conn_ids += 1

    def disconnect(self, conn):
        print('disconnected')
        self.conns.pop(conn, None)

    def _send_thread(self):
        while not self.shutdown_flag:
            try:
                conn, send_data = self.output_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                conn.send(send_data)
            except BrokenPipeError:
                self.disconnect(conn)

    def _recv_thread(self):
        while not self.shutdown_flag:
            conn_list, _, _ = select.select(self.conns, [], [], 0.3)
            for conn in conn_list:
                try:
                    recv_data = conn.recv()
                except ConnectionResetError:
                    self.disconnect(conn)
                    continue
                except EOFError:
                    self.disconnect(conn)
                    continue
                while not self.shutdown_flag:
                    try:
                        self.input_queue.put((conn, recv_data), timeout=0.3)
                        break
                    except queue.Full:
                        pass
