import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "slurm" / "train-gaitlu-study.sbatch"


class TrainGaitLUStudySlurmTest(unittest.TestCase):
    def _environment(self, root: Path) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "CODY_JEPA_ROOT": str(PROJECT_ROOT),
                "GAITLU_PREPARED_ROOT": str(root / "prepared"),
                "CODY_JEPA_RUN_ROOT": str(root / "runs"),
            }
        )
        return environment

    def test_launcher_submits_three_sequential_groups_of_at_most_eight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary_root = root / "bin"
            binary_root.mkdir()
            log_path = root / "sbatch.log"
            fake_sbatch = binary_root / "sbatch"
            fake_sbatch.write_text(
                "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$SBATCH_TEST_LOG\"\n",
                encoding="utf-8",
            )
            fake_sbatch.chmod(0o755)
            environment = self._environment(root)
            environment["PATH"] = f"{binary_root}:{environment['PATH']}"
            environment["SBATCH_TEST_LOG"] = str(log_path)

            subprocess.run(["bash", str(SCRIPT)], env=environment, check=True)

            submissions = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(submissions), 3)
            for submission, expected_range in zip(
                submissions, ("0-7%8", "8-15%8", "16-19%8")
            ):
                self.assertIn("--wait", submission)
                self.assertIn(f"--array={expected_range}", submission)
                self.assertIn("--export=ALL", submission)
                self.assertTrue(submission.endswith(str(SCRIPT)))

    def test_array_worker_trains_only_its_registered_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary_root = root / "bin"
            binary_root.mkdir()
            log_path = root / "uv.log"
            for command in ("nvidia-smi", "uv"):
                executable = binary_root / command
                executable.write_text(
                    "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$UV_TEST_LOG\"\n",
                    encoding="utf-8",
                )
                executable.chmod(0o755)
            environment = self._environment(root)
            environment["PATH"] = f"{binary_root}:{environment['PATH']}"
            environment["UV_TEST_LOG"] = str(log_path)
            environment["SLURM_ARRAY_TASK_ID"] = "17"

            subprocess.run(["bash", str(SCRIPT)], env=environment, check=True)

            invocation = log_path.read_text(encoding="utf-8")
            self.assertIn("cody-jepa-train-gaitlu-study", invocation)
            self.assertIn(str(root / "prepared" / "training_registry.csv"), invocation)
            self.assertIn("--run-index 17", invocation)
            self.assertIn(f"--output-root {root / 'runs'}", invocation)
            self.assertIn("--device cuda", invocation)

    def test_launcher_stops_after_a_failed_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary_root = root / "bin"
            binary_root.mkdir()
            log_path = root / "sbatch.log"
            count_path = root / "sbatch.count"
            fake_sbatch = binary_root / "sbatch"
            fake_sbatch.write_text(
                "#!/bin/bash\n"
                "count=$(($(cat \"$SBATCH_TEST_COUNT\" 2>/dev/null || echo 0) + 1))\n"
                "printf '%s\\n' \"$count\" > \"$SBATCH_TEST_COUNT\"\n"
                "printf '%s\\n' \"$*\" >> \"$SBATCH_TEST_LOG\"\n"
                "((count != 2))\n",
                encoding="utf-8",
            )
            fake_sbatch.chmod(0o755)
            environment = self._environment(root)
            environment["PATH"] = f"{binary_root}:{environment['PATH']}"
            environment["SBATCH_TEST_LOG"] = str(log_path)
            environment["SBATCH_TEST_COUNT"] = str(count_path)

            result = subprocess.run(["bash", str(SCRIPT)], env=environment, check=False)

            self.assertNotEqual(result.returncode, 0)
            submissions = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(submissions), 2)
            self.assertIn("--array=0-7%8", submissions[0])
            self.assertIn("--array=8-15%8", submissions[1])


if __name__ == "__main__":
    unittest.main()
