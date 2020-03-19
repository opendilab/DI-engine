import socket
import os


def get_ip():
    # beware: return 127.0.0.1 on some slurm nodes
    myname = socket.getfqdn(socket.gethostname())
    myaddr = socket.gethostbyname(myname)
    return myaddr


def get_pid():
    return os.getpid()


def get_actor_id():
    return os.getenv('SLURM_JOB_ID', 'PID' + str(get_pid()))


if __name__ == "__main__":
    print('ip', get_ip())
    print('pid', get_pid())
