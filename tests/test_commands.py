from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CommandCompatibilityTests(unittest.TestCase):
    def test_cli_and_server_scripts_work_outside_the_project_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for script in (
                "cli/hybrid_search_cli.py",
                "cli/keyword_search_cli.py",
                "cli/semantic_search_cli.py",
            ):
                with self.subTest(script=script):
                    result = subprocess.run(
                        [sys.executable, str(PROJECT_ROOT / script), "--help"],
                        cwd=directory, capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_package_and_original_cli_return_the_same_normalisation(self):
        outputs = []
        for entry in (["-m", "cli.hybrid_search_cli"], ["cli/hybrid_search_cli.py"]):
            result = subprocess.run(
                [sys.executable, *entry, "normalize", "1", "2", "3"],
                cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append(result.stdout)
        self.assertEqual(outputs, ["* 0.0000\n* 0.5000\n* 1.0000\n"] * 2)
