import json
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class UvWorkflowPolicyTest(unittest.TestCase):
    def test_maintained_docs_and_scripts_do_not_use_other_environment_managers(self):
        paths = [PROJECT_ROOT / "README.md"]
        paths.extend(sorted((PROJECT_ROOT / "tutorials").rglob("*.md")))
        paths.extend(sorted((PROJECT_ROOT / "notes").rglob("*.md")))
        paths.extend(sorted((PROJECT_ROOT / "slurm").glob("*.sbatch")))
        forbidden = re.compile(
            r"(?im)^\s*(?:python(?:3)?|jupyter|pip(?:3)?|conda|poetry)(?:\s|$)"
            r"|\b(?:pip(?:3)? install|conda install|poetry add)\b"
        )
        violations = []
        for path in paths:
            for match in forbidden.finditer(path.read_text()):
                line = path.read_text().count("\n", 0, match.start()) + 1
                violations.append(f"{path.relative_to(PROJECT_ROOT)}:{line}")
        self.assertEqual(violations, [], f"non-uv commands found: {violations}")

    def test_source_notebooks_do_not_bypass_uv_for_shell_commands(self):
        forbidden = re.compile(
            r"(?im)^\s*(?:!|%)(?:python(?:3)?|jupyter|pip(?:3)?|conda|poetry)\b"
        )
        violations = []
        paths = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))
        paths.append(PROJECT_ROOT / "haic-results" / "job_91108.ipynb")
        for path in paths:
            notebook = json.loads(path.read_text())
            source = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
            )
            if forbidden.search(source):
                violations.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(violations, [], f"non-uv notebook commands found: {violations}")


if __name__ == "__main__":
    unittest.main()
