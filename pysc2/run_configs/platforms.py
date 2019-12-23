# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configs for how to run SC2 from a normal install on various platforms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from absl import logging
import os
import platform
import subprocess
import sys

from absl import flags

from pysc2.lib import sc_process
from pysc2.run_configs import lib

# https://github.com/Blizzard/s2client-proto/blob/master/buildinfo/versions.json
# Generate with bin/gen_versions.py
VERSIONS = {ver.game_version: ver for ver in [
    lib.Version("3.16.1", 55958, "5BD7C31B44525DAB46E64C4602A81DC2", None),
    lib.Version("3.17.0", 56787, "DFD1F6607F2CF19CB4E1C996B2563D9B", None),
    lib.Version("3.17.1", 56787, "3F2FCED08798D83B873B5543BEFA6C4B", None),
    lib.Version("3.17.2", 56787, "C690FC543082D35EA0AAA876B8362BEA", None),
    lib.Version("3.18.0", 57507, "1659EF34997DA3470FF84A14431E3A86", None),
    lib.Version("3.19.0", 58400, "2B06AEE58017A7DF2A3D452D733F1019", None),
    lib.Version("3.19.1", 58400, "D9B568472880CC4719D1B698C0D86984", None),
    lib.Version("4.0.0", 59587, "9B4FD995C61664831192B7DA46F8C1A1", None),
    lib.Version("4.0.2", 59587, "B43D9EE00A363DAFAD46914E3E4AF362", None),
    lib.Version("4.1.0", 60196, "1B8ACAB0C663D5510941A9871B3E9FBE", None),
    lib.Version("4.1.1", 60321, "5C021D8A549F4A776EE9E9C1748FFBBC", None),
    lib.Version("4.1.2", 60321, "33D9FE28909573253B7FC352CE7AEA40", None),
    lib.Version("4.2.0", 62347, "C0C0E9D37FCDBC437CE386C6BE2D1F93", None),
    lib.Version("4.2.1", 62848, "29BBAC5AFF364B6101B661DB468E3A37", None),
    lib.Version("4.2.2", 63454, "3CB54C86777E78557C984AB1CF3494A0", None),
    lib.Version("4.3.0", 64469, "C92B3E9683D5A59E08FC011F4BE167FF", None),
    lib.Version("4.3.1", 65094, "E5A21037AA7A25C03AC441515F4E0644", None),
    lib.Version("4.3.2", 65384, "B6D73C85DFB70F5D01DEABB2517BF11C", None),
    lib.Version("4.4.0", 65895, "BF41339C22AE2EDEBEEADC8C75028F7D", None),
    lib.Version("4.4.1", 66668, "C094081D274A39219061182DBFD7840F", None),
    lib.Version("4.5.0", 67188, "2ACF84A7ECBB536F51FC3F734EC3019F", None),
    lib.Version("4.5.1", 67188, "6D239173B8712461E6A7C644A5539369", None),
    lib.Version("4.6.0", 67926, "7DE59231CBF06F1ECE9A25A27964D4AE", None),
    lib.Version("4.6.1", 67926, "BEA99B4A8E7B41E62ADC06D194801BAB", None),
    lib.Version("4.6.2", 69232, "B3E14058F1083913B80C20993AC965DB", None),
    lib.Version("4.7.0", 70154, "8E216E34BC61ABDE16A59A672ACB0F3B", None),
    lib.Version("4.7.1", 70154, "94596A85191583AD2EBFAE28C5D532DB", None),
    lib.Version("4.8.0", 71061, "760581629FC458A1937A05ED8388725B", None),
    lib.Version("4.8.1", 71523, "FCAF3F050B7C0CC7ADCF551B61B9B91E", None),
    lib.Version("4.8.2", 71663, "FE90C92716FC6F8F04B74268EC369FA5", None),
    lib.Version("4.8.3", 72282, "0F14399BBD0BA528355FF4A8211F845B", None),
    lib.Version("4.8.4", 73286, "CD040C0675FD986ED37A4CA3C88C8EB5", None),
]}

flags.DEFINE_enum("sc2_version", None, sorted(lib.VERSIONS.keys()),
                  "Which version of the game to use.")
flags.DEFINE_bool("sc2_dev_build", False,
                  "Use a dev build. Mostly useful for testing by Blizzard.")
FLAGS = flags.FLAGS


def _read_execute_info(path, parents):
    """Read the ExecuteInfo.txt file and return the base directory."""
    path = os.path.join(path, "StarCraft II/ExecuteInfo.txt")
    if os.path.exists(path):
        with open(path, "rb") as f:  # Binary because the game appends a '\0' :(.
            for line in f:
                parts = [p.strip() for p in line.decode("utf-8").split("=")]
                if len(parts) == 2 and parts[0] == "executable":
                    exec_path = parts[1].replace("\\", "/")  # For windows compatibility.
                    for _ in range(parents):
                        exec_path = os.path.dirname(exec_path)
                    return exec_path


class LocalBase(lib.RunConfig):
    """Base run config for public installs."""

    def __init__(self, base_dir, exec_name, version, cwd=None, env=None):
        base_dir = os.path.expanduser(base_dir)
        version = version or FLAGS.sc2_version or "latest"
        cwd = cwd and os.path.join(base_dir, cwd)
        super(LocalBase, self).__init__(
            replay_dir=os.path.join(base_dir, "Replays"),
            data_dir=base_dir, tmp_dir=None, version=version, cwd=cwd, env=env)
        if FLAGS.sc2_dev_build:
            self.version = self.version._replace(build_version=0)
        elif self.version.build_version < lib.VERSIONS["3.16.1"].build_version:
            raise sc_process.SC2LaunchError(
                "SC2 Binaries older than 3.16.1 don't support the api.")
        self._exec_name = exec_name

    def start(self, want_rgb=True, **kwargs):
        """Launch the game."""
        del want_rgb  # Unused
        if not os.path.isdir(self.data_dir):
            raise sc_process.SC2LaunchError(
                "Expected to find StarCraft II installed at '%s'. If it's not "
                "installed, do that and run it once so auto-detection works. If "
                "auto-detection failed repeatedly, then set the SC2PATH environment "
                "variable with the correct location." % self.data_dir)
        exec_path = os.path.join(
            self.data_dir, "Versions/Base%05d" % self.version.build_version,
            self._exec_name)

        if not os.path.exists(exec_path):
            raise sc_process.SC2LaunchError("No SC2 binary found at: %s" % exec_path)

        return sc_process.StarcraftProcess(
            self, exec_path=exec_path, version=self.version, **kwargs)

    def get_versions(self, containing=None):
        versions_dir = os.path.join(self.data_dir, "Versions")
        version_prefix = "Base"
        versions_found = sorted(int(v[len(version_prefix):])
                                for v in os.listdir(versions_dir)
                                if v.startswith(version_prefix))
        if not versions_found:
            raise sc_process.SC2LaunchError(
                "No SC2 Versions found in %s" % versions_dir)
        known_versions = [v for v in lib.VERSIONS.values()
                          if v.build_version in versions_found]
        # Add one more with the max version. That one doesn't need a data version
        # since SC2 will find it in the .build.info file. This allows running
        # versions newer than what are known by pysc2, and so is the default.
        known_versions.append(
            lib.Version("latest", max(versions_found), None, None))
        ret = lib.version_dict(known_versions)
        if containing is not None and containing not in ret:
            raise ValueError("Unknown game version: %s. Known versions: %s." % (
                containing, sorted(ret.keys())))
        return ret


class Windows(LocalBase):
    """Run on Windows."""

    def __init__(self, version=None):
        exec_path = (os.environ.get("SC2PATH") or
                     _read_execute_info(os.path.expanduser("~/Documents"), 3) or
                     "C:/Program Files (x86)/StarCraft II")
        super(Windows, self).__init__(exec_path, "SC2_x64.exe",
                                      version=version, cwd="Support64")

    @classmethod
    def priority(cls):
        if platform.system() == "Windows":
            return 1


class Cygwin(LocalBase):
    """Run on Cygwin. This runs the windows binary within a cygwin terminal."""

    def __init__(self, version=None):
        exec_path = os.environ.get(
            "SC2PATH", "/cygdrive/c/Program Files (x86)/StarCraft II")
        super(Cygwin, self).__init__(exec_path, "SC2_x64.exe",
                                     version=version, cwd="Support64")

    @classmethod
    def priority(cls):
        if sys.platform == "cygwin":
            return 1


class MacOS(LocalBase):
    """Run on MacOS."""

    def __init__(self, version=None):
        exec_path = (os.environ.get("SC2PATH") or
                     _read_execute_info(os.path.expanduser(
                         "~/Library/Application Support/Blizzard"), 6) or
                     "/Applications/StarCraft II")
        super(MacOS, self).__init__(exec_path, "SC2.app/Contents/MacOS/SC2",
                                    version=version)

    @classmethod
    def priority(cls):
        if platform.system() == "Darwin":
            return 1


class Linux(LocalBase):
    """Config to run on Linux."""

    known_gl_libs = [  # In priority order. Prefer hardware rendering.
        ("-eglpath", "libEGL.so"),
        ("-eglpath", "libEGL.so.1"),
        ("-osmesapath", "libOSMesa.so"),
        ("-osmesapath", "libOSMesa.so.8"),  # Ubuntu 16.04
        ("-osmesapath", "libOSMesa.so.6"),  # Ubuntu 14.04
    ]

    def __init__(self, version=None):
        base_dir = os.environ.get("SC2PATH", "~/StarCraftII")
        base_dir = os.path.expanduser(base_dir)
        env = copy.deepcopy(os.environ)
        env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
            os.environ.get("LD_LIBRARY_PATH"),
            os.path.join(base_dir, "Libs/")]))
        super(Linux, self).__init__(base_dir, "SC2_x64", version=version, env=env)

    @classmethod
    def priority(cls):
        if platform.system() == "Linux":
            return 1

    def start(self, want_rgb=True, **kwargs):
        extra_args = kwargs.pop("extra_args", [])

        if want_rgb:
            # Figure out whether the various GL libraries exist since SC2 sometimes
            # fails if you ask to use a library that doesn't exist.
            libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
            libs = {lib.strip().split()[0] for lib in libs.split("\n") if lib}
            for arg, lib_name in self.known_gl_libs:
                if lib_name in libs:
                    extra_args += [arg, lib_name]
                    break
            else:
                extra_args += ["-headlessNoRender"]
                logging.info(
                    "No GL library found, so RGB rendering will be disabled. "
                    "For software rendering install libosmesa.")

        return super(Linux, self).start(
            want_rgb=want_rgb, extra_args=extra_args, **kwargs)
