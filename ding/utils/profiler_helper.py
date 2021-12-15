import atexit
import pstats
import io
import cProfile
import os


def register_profiler(write_profile, pr, folder_path):
    atexit.register(write_profile, pr, folder_path)


class Profiler:

    def __init__(self):
        self.pr = cProfile.Profile()

    def mkdir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def write_profile(self, pr, folder_path):
        pr.disable()
        s_tottime = io.StringIO()
        s_cumtime = io.StringIO()

        ps = pstats.Stats(pr, stream=s_tottime).sort_stats('tottime')
        ps.print_stats()
        with open(folder_path + "/profile_tottime.txt", 'w+') as f:
            f.write(s_tottime.getvalue())

        ps = pstats.Stats(pr, stream=s_cumtime).sort_stats('cumtime')
        ps.print_stats()
        with open(folder_path + "/profile_cumtime.txt", 'w+') as f:
            f.write(s_cumtime.getvalue())

        pr.dump_stats(folder_path + "/profile.prof")

    def profile(self, folder_path="./tmp"):
        self.mkdir(folder_path)
        self.pr.enable()
        register_profiler(self.write_profile, self.pr, folder_path)
