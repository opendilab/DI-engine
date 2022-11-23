import os
from subprocess import Popen, PIPE


def get_n_values(seed, N=100):
    """
    Generates the same random numbers as the lux ai CLI tool given the numerical seed and a number of values to generate. 10k values is generally more than enough
    """
    p = Popen(["node", f"{os.path.dirname(__file__)}/rng.js", str(seed), str(N)], stdout=PIPE)
    # p.stdin.write(str(seed).encode())
    output = p.stdout.readline()
    vals = [float(v) for v in output.decode().split(",")]
    p.stdout.close()
    p.kill()
    p.wait()
    return vals
