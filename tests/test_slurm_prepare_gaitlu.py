import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "slurm" / "prepare-gaitlu-shards.sbatch"


class PrepareGaitLUShardsSlurmTest(unittest.TestCase):
    def _environment(self, root: Path) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "CODY_JEPA_ROOT": str(PROJECT_ROOT),
                "GAITLU_RAW_ROOT": str(root / "raw"),
                "GAITLU_PREPARED_ROOT": str(root / "prepared"),
                "PREP_LOG_ROOT": str(root / "logs" / "prepare-shards"),
            }
        )
        return environment

    def test_launcher_submits_at_most_eight_shards_and_waits_between_groups(self):
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
            self.assertEqual(len(submissions), 13)
            preparation_logs = Path(environment["PREP_LOG_ROOT"])
            self.assertTrue(preparation_logs.is_dir())
            expected_ranges = [
                f"{start}-{min(start + 7, 99)}%8" for start in range(0, 100, 8)
            ]
            for submission, expected_range in zip(submissions, expected_ranges):
                self.assertIn("--wait", submission)
                self.assertIn(f"--array={expected_range}", submission)
                self.assertIn("--export=ALL", submission)
                self.assertIn(
                    f"--output={preparation_logs}/slurm-%x-%A_%a.out", submission
                )
                self.assertTrue(submission.endswith(str(SCRIPT)))

    def test_array_worker_converts_only_its_assigned_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary_root = root / "bin"
            binary_root.mkdir()
            log_path = root / "uv.log"
            fake_uv = binary_root / "uv"
            fake_uv.write_text(
                "#!/bin/bash\nprintf '%s\\n' \"$*\" >> \"$UV_TEST_LOG\"\n",
                encoding="utf-8",
            )
            fake_uv.chmod(0o755)
            environment = self._environment(root)
            environment["PATH"] = f"{binary_root}:{environment['PATH']}"
            environment["UV_TEST_LOG"] = str(log_path)
            environment["SLURM_ARRAY_TASK_ID"] = "9"

            subprocess.run(["bash", str(SCRIPT)], env=environment, check=True)

            invocation = log_path.read_text(encoding="utf-8").strip()
            self.assertIn("cody-jepa-prepare-gaitlu pack-shard", invocation)
            self.assertIn(str(root / "raw" / "gaitlu-009.tar.gz"), invocation)
            self.assertIn(str(root / "prepared"), invocation)


if __name__ == "__main__":
    unittest.main()
