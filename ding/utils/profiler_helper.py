import atexit
import pstats
import io
import cProfile
import os


def register_profiler(write_profile, pr, folder_path):
    atexit.register(write_profile, pr, folder_path)


class Profiler:
    """
    Overview:
        A class for profiling code execution. It can be used as a context manager or a decorator.

    Interfaces:
        ``__init__``, ``mkdir``, ``write_profile``, ``profile``.
    """

    def __init__(self):
        """
        Overview:
            Initialize the Profiler object.
        """

        self.pr = cProfile.Profile()

    def mkdir(self, directory: str):
        """
        OverView:
            Create a directory if it doesn't exist.

        Arguments:
            - directory (:obj:`str`): The path of the directory to be created.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def write_profile(self, pr: cProfile.Profile, folder_path: str):
        """
        OverView:
            Write the profiling results to files.

        Arguments:
            - pr (:obj:`cProfile.Profile`): The profiler object containing the profiling results.
            - folder_path (:obj:`str`): The path of the folder where the profiling files will be saved.
        """
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
        """
        OverView:
            Enable profiling and save the results to files.

        Arguments:
            - folder_path (:obj:`str`): The path of the folder where the profiling files will be saved. \
                Defaults to "./tmp".
        """
        self.mkdir(folder_path)
        self.pr.enable()
        register_profiler(self.write_profile, self.pr, folder_path)
