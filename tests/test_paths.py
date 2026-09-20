"""Where the installed app keeps its files (it must never write next to its own executable)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eyemouse.config import find_model, resolve_data_dir

ROOT = Path("C:/project")


class DataDirTest(unittest.TestCase):
    def test_development_uses_the_project_data_folder(self):
        self.assertEqual(resolve_data_dir(False, {}, ROOT), ROOT / "data")

    def test_installed_app_uses_appdata(self):
        self.assertEqual(resolve_data_dir(True, {"APPDATA": "C:/Users/x/AppData/Roaming"}, ROOT),
                         Path("C:/Users/x/AppData/Roaming") / "EyeMouse")

    def test_environment_override_wins_in_both_modes(self):
        env = {"EYEMOUSE_DATA_DIR": "D:/portable", "APPDATA": "C:/ignored"}
        self.assertEqual(resolve_data_dir(True, env, ROOT), Path("D:/portable"))
        self.assertEqual(resolve_data_dir(False, env, ROOT), Path("D:/portable"))


class ModelLookupTest(unittest.TestCase):
    def test_bundled_model_is_preferred_and_data_dir_is_the_download_target(self):
        with tempfile.TemporaryDirectory() as d:
            res, data = Path(d) / "res", Path(d) / "data"
            self.assertEqual(find_model("m.task", data, res), data / "m.task")      # nothing bundled: download target
            (res / "models").mkdir(parents=True)
            (res / "models" / "m.task").write_bytes(b"x")
            self.assertEqual(find_model("m.task", data, res), res / "models" / "m.task")


if __name__ == "__main__":
    unittest.main()
