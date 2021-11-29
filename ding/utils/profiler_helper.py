import atexit
import pstats
import io
import cProfile
import os


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def profiler(file_path="tmp/profile.txt"):
    mkdir(os.path.dirname(file_path))
    pr = cProfile.Profile()
    pr.enable()

    def write_profile():
        pr.disable()
        s = io.StringIO()

        ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
        ps.print_stats()
        with open(file_path, 'w') as f:
            f.write(s.getvalue())

        ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
        ps.print_stats()
        with open(file_path, 'w+') as f:
            f.write(s.getvalue())

    atexit.register(write_profile)
