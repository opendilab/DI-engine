import socket
import os


def get_ip():
    myname = socket.getfqdn(socket.gethostname())
    myaddr = socket.gethostbyname(myname)
    return myaddr


def get_pid():
    return os.getpid()


if __name__ == "__main__":
    print('ip', get_ip())
    print('pid', get_pid())
