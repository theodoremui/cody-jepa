import json
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAINTAINED_NOTEBOOKS = tuple(sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")))
RETAINED_EVIDENCE_NOTEBOOK = PROJECT_ROOT / "haic-results" / "job_91108.ipynb"
FORBIDDEN_NOTEBOOK_COMMAND = re.compile(
    r"(?im)^\s*(?:!|%)\s*(?:python(?:3)?|jupyter|pip(?:3)?|conda|poetry)\b"
    r"|^\s*!\s*(?:uv\s+pip(?:\s+install|\s+uninstall|\s+sync)|uvx(?:\s|$))"
)


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )


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

    def test_maintained_source_notebooks_document_uv_workflow(self):
        violations = []
        uv_jupyter = re.compile(r"\buv run(?:\s+--[\w-]+)*\s+jupyter\b")
        for path in MAINTAINED_NOTEBOOKS:
            source = notebook_source(path)
            missing = []
            if not uv_jupyter.search(source):
                missing.append("an explicit uv run jupyter launch")
            for command in ("uv sync", "uv add", "uv remove", "uv lock"):
                if command not in source:
                    missing.append(command)
            if missing:
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}: missing {', '.join(missing)}"
                )
        self.assertEqual(
            violations,
            [],
            "maintained notebooks must document the uv-only workflow:\n"
            + "\n".join(violations),
        )

    def test_source_notebooks_do_not_bypass_uv_for_shell_commands(self):
        violations = []
        for path in (*MAINTAINED_NOTEBOOKS, RETAINED_EVIDENCE_NOTEBOOK):
            if FORBIDDEN_NOTEBOOK_COMMAND.search(notebook_source(path)):
                violations.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(violations, [], f"non-uv notebook commands found: {violations}")

    def test_notebook_command_policy_rejects_spacing_and_uv_install_bypasses(self):
        forbidden = (
            "%pip install package",
            "% pip install package",
            "!pip install package",
            "! pip install package",
            "!uv pip install package",
            "!uv pip uninstall package",
            "!uv pip sync requirements.txt",
            "!uvx tool",
        )
        for command in forbidden:
            with self.subTest(command=command):
                self.assertRegex(command, FORBIDDEN_NOTEBOOK_COMMAND)

        allowed = (
            "!uv sync --frozen",
            "!uv run python scripts/check.py",
            "uv add package",
        )
        for command in allowed:
            with self.subTest(command=command):
                self.assertNotRegex(command, FORBIDDEN_NOTEBOOK_COMMAND)

    def test_readme_and_haic_guide_document_uv_jupyter_and_dependency_policy(self):
        expected = {
            PROJECT_ROOT / "README.md": (
                "uv run jupyter lab notebooks/healthgait_manifest_loader.ipynb",
                "uv run jupyter lab notebooks/single-stream-jepa.ipynb",
                "uv run jupyter lab notebooks/linear-probe-results.ipynb",
            ),
            PROJECT_ROOT / "tutorials" / "haic-guide.md": (
                "uv run jupyter lab notebooks/single-stream-jepa.ipynb",
            ),
        }
        violations = []
        for path, launches in expected.items():
            source = path.read_text()
            missing = [launch for launch in launches if launch not in source]
            missing.extend(
                command
                for command in ("uv sync", "uv add", "uv remove", "uv lock")
                if command not in source
            )
            if missing:
                violations.append(
                    f"{path.relative_to(PROJECT_ROOT)}: missing {', '.join(missing)}"
                )
        self.assertEqual(violations, [], "incomplete uv documentation:\n" + "\n".join(violations))


if __name__ == "__main__":
    unittest.main()
