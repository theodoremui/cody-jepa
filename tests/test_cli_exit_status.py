import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
SOURCE_ROOT = PROJECT_ROOT / "src"


def _console_targets() -> list[tuple[str, str]]:
    contents = PYPROJECT.read_text(encoding="utf-8")
    scripts_section = contents.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    return re.findall(r'^\S+\s*=\s*"([^:"]+):([^\"]+)"$', scripts_section, re.MULTILINE)


def _explicit_return_values(function: ast.FunctionDef) -> list[ast.Return]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(function):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    returns = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        ancestor = parents.get(node)
        while ancestor is not None and ancestor is not function:
            if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                break
            ancestor = parents.get(ancestor)
        else:
            returns.append(node)
    return returns


class ConsoleExitStatusTest(unittest.TestCase):
    def test_console_main_functions_do_not_return_application_results(self):
        failures = []
        for module_name, function_name in _console_targets():
            path = SOURCE_ROOT.joinpath(*module_name.split(".")).with_suffix(".py")
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            functions = [
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == function_name
            ]
            # A few compatibility modules re-export an implementation from elsewhere.
            # Their implementation is covered in its owning module's tests.
            if not functions:
                continue
            self.assertEqual(len(functions), 1, f"expected one {module_name}:{function_name}")
            for returned in _explicit_return_values(functions[0]):
                failures.append(f"{path.relative_to(PROJECT_ROOT)}:{returned.lineno}")

        self.assertEqual(
            failures,
            [],
            "console main functions are passed to sys.exit by installed wrappers and must "
            "return None; move application results into a separate function: "
            + ", ".join(failures),
        )

    def test_prepare_gaitlu_success_exits_zero_and_prints_result_once(self):
        from cody_jepa.cli.prepare_gaitlu import main

        result = {"valid_sequences": 3, "excluded_sequences": 1}
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("cody_jepa.cli.prepare_gaitlu.pack_gaitlu_shard", return_value=result),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            sys.exit(
                main(
                    [
                        "pack-shard",
                        "--input",
                        "gaitlu-000.tar.gz",
                        "--prepared-root",
                        "prepared",
                        "--trust-pickles",
                    ]
                )
            )

        self.assertIsNone(raised.exception.code)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(json.loads(stdout.getvalue()), result)

    def test_train_success_exits_zero(self):
        from cody_jepa.cli.train import main

        with (
            patch("cody_jepa.cli.train.run_training", return_value={"global_step": 1}) as run,
            self.assertRaises(SystemExit) as raised,
        ):
            sys.exit(main(["--config", "config.json", "--output-dir", "output"]))

        self.assertIsNone(raised.exception.code)
        run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
