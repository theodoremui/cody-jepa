"""Fast structural validation for the tutorial curriculum.

This script uses only the Python standard library. It checks file coverage,
notebook structure, local links, GitHub-safe lecture math, visual coverage, and
a few project style constraints.
Notebook execution is intentionally handled by nbconvert, as documented in the
tutorial README.
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from ast import Import, ImportFrom, parse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LECTURES = ROOT / "lectures"
IMPLEMENTATIONS = ROOT / "implementations"
IMAGES = ROOT / "images"
EXPECTED_BASENAMES = {
    "01_spatiotemporal_tensor_geometry",
    "02_inner_product_geometry",
    "03_hierarchical_observations",
    "04_attention_and_positions",
    "05_masked_latent_prediction",
    "06_representation_collapse",
    "07_gradient_updates_and_schedules",
    "08_group_aware_sampling",
    "09_eigenspectra_and_effective_rank",
    "10_regularized_linear_estimation",
    "11_factorial_state_spaces",
    "12_blockwise_distances_and_ranking",
    "13_context_interventions",
    "14_paired_inference",
    "15_exposure_and_replication",
    "16_reproducible_scientific_evaluators",
}
ALLOWED_THIRD_PARTY_IMPORTS = {
    "IPython",
    "matplotlib",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
}


def validate_tex_expression(
    expression: str, location: str, errors: list[str]
) -> None:
    depth = 0
    for index, character in enumerate(expression):
        escaped = index > 0 and expression[index - 1] == "\\"
        if escaped:
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                errors.append(f"{location} closes a TeX group that was not opened")
                break
    if depth > 0:
        errors.append(f"{location} has {depth} unclosed TeX group(s)")

    environment_stack: list[str] = []
    for action, environment in re.findall(
        r"\\(begin|end)\{([^}]+)\}", expression
    ):
        if action == "begin":
            environment_stack.append(environment)
        elif not environment_stack or environment_stack.pop() != environment:
            errors.append(f"{location} has mismatched TeX environments")
            environment_stack.clear()
            break
    if environment_stack:
        errors.append(f"{location} has unclosed TeX environments")


def collect_basenames(directory: Path, suffix: str) -> set[str]:
    return {path.stem for path in directory.glob(f"[0-9][0-9]_*.{suffix}")}


def validate_notebook(path: Path, errors: list[str]) -> None:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path.relative_to(ROOT)} is not valid JSON: {exc}")
        return

    if notebook.get("nbformat") != 4:
        errors.append(f"{path.relative_to(ROOT)} must use notebook format 4")

    cells = notebook.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    if len(code_cells) < 3 or len(markdown_cells) < 3:
        errors.append(
            f"{path.relative_to(ROOT)} needs at least three code and three markdown cells"
        )
    if not 8 <= len(cells) <= 20:
        errors.append(f"{path.relative_to(ROOT)} should contain 8 to 20 focused cells")

    for cell in code_cells:
        if not isinstance(cell.get("source", []), list):
            errors.append(f"{path.relative_to(ROOT)} has a code cell with non-list source")
        if cell.get("execution_count") is not None:
            errors.append(f"{path.relative_to(ROOT)} contains stored execution counts")
        if cell.get("outputs"):
            errors.append(f"{path.relative_to(ROOT)} contains stored outputs")

    for cell in cells:
        if not cell.get("id"):
            errors.append(f"{path.relative_to(ROOT)} has a cell without a stable id")

    kernel_name = notebook.get("metadata", {}).get("kernelspec", {}).get("name")
    if kernel_name not in {"python3", "python"}:
        errors.append(f"{path.relative_to(ROOT)} must declare a Python kernel")

    code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    if "assert " not in code:
        errors.append(f"{path.relative_to(ROOT)} must assert at least one invariant")
    if "seed" not in code.lower():
        errors.append(f"{path.relative_to(ROOT)} must set a deterministic seed")
    if "cody_jepa" in code:
        errors.append(f"{path.relative_to(ROOT)} must not depend on repository modules")

    try:
        tree = parse(code)
    except SyntaxError as exc:
        errors.append(f"{path.relative_to(ROOT)} code cannot be parsed: {exc}")
        return
    for node in tree.body:
        modules: list[str] = []
        if isinstance(node, Import):
            modules = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ImportFrom) and node.module:
            modules = [node.module.split(".")[0]]
        for module in modules:
            if module not in sys.stdlib_module_names and module not in ALLOWED_THIRD_PARTY_IMPORTS:
                errors.append(
                    f"{path.relative_to(ROOT)} imports undeclared dependency {module}"
                )

    markdown = "\n".join("".join(cell.get("source", [])) for cell in markdown_cells)
    expected_image = f"../images/{path.stem}.svg"
    if expected_image not in markdown:
        errors.append(f"{path.relative_to(ROOT)} must embed {expected_image}")
    expected_lecture = f"../lectures/{path.stem}.md"
    if expected_lecture not in markdown:
        errors.append(f"{path.relative_to(ROOT)} must link back to {expected_lecture}")
    if "../README.md" not in markdown:
        errors.append(f"{path.relative_to(ROOT)} must link back to ../README.md")


def validate_lecture(path: Path, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    if len(text.splitlines()) < 100:
        errors.append(f"{path.relative_to(ROOT)} is too short for a detailed lecture")
    word_count = len(re.findall(r"\b[\w'-]+\b", text))
    if word_count < 2000:
        errors.append(
            f"{path.relative_to(ROOT)} needs at least 2,000 words of detailed teaching"
        )
    required_ideas = {
        "learning goals": "learning goals",
        "efficiency notes": "efficien",
        "failure modes": "failure",
        "exercises": "exercise",
    }
    lowered = text.lower()
    for label, marker in required_ideas.items():
        if marker not in lowered:
            errors.append(f"{path.relative_to(ROOT)} is missing {label}")
    expected_image = f"../images/{path.stem}.svg"
    if expected_image not in text:
        errors.append(f"{path.relative_to(ROOT)} must embed {expected_image}")
    expected_notebook = f"../implementations/{path.stem}.ipynb"
    if expected_notebook not in text:
        errors.append(f"{path.relative_to(ROOT)} must link to {expected_notebook}")

    image_targets = re.findall(r"!\[[^\]]+\]\(([^)]+\.svg)\)", text)
    if len(image_targets) < 3:
        errors.append(
            f"{path.relative_to(ROOT)} needs at least three explanatory SVG figures"
        )

    forbidden_operator_macro = "\\" + "operator" + "name"
    forbidden_math = {
        r"\(": r"\(...\) inline delimiters",
        r"\)": r"\(...\) inline delimiters",
        r"\[": r"\[...\] display delimiters",
        r"\]": r"\[...\] display delimiters",
        forbidden_operator_macro: "the forbidden operator-name macro",
    }
    for marker, label in forbidden_math.items():
        if marker in text:
            errors.append(
                f"{path.relative_to(ROOT)} contains forbidden GitHub math syntax {label}"
            )

    display_fences = 0
    display_open = False
    lecture_lines = text.splitlines()
    for line_number, line in enumerate(lecture_lines, start=1):
        if display_open and re.fullmatch(r"[=-]+", line.strip()):
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number} contains a Setext-like "
                "line inside display math; join it to an adjacent TeX line or use "
                "a GitHub-safe relation such as {}={}"
            )
        if "$$" in line:
            if line.strip() != "$$":
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_number} must put $$ on its own line"
                )
            else:
                display_fences += 1
                line_index = line_number - 1
                if not display_open:
                    if line_index > 0 and lecture_lines[line_index - 1].strip():
                        errors.append(
                            f"{path.relative_to(ROOT)}:{line_number} needs a blank line before $$"
                        )
                    display_open = True
                else:
                    if (
                        line_index + 1 < len(lecture_lines)
                        and lecture_lines[line_index + 1].strip()
                    ):
                        errors.append(
                            f"{path.relative_to(ROOT)}:{line_number} needs a blank line after $$"
                        )
                    display_open = False
    if display_fences % 2:
        errors.append(f"{path.relative_to(ROOT)} has an unmatched $$ display delimiter")
    if display_fences < 4:
        errors.append(
            f"{path.relative_to(ROOT)} needs at least two interpreted display equations"
        )

    math_text = re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.DOTALL)
    math_text = re.sub(r"`[^`\n]*`", "", math_text)
    display_pattern = re.compile(
        r"(?ms)^\$\$\s*$\n(.*?)\n^\$\$\s*$"
    )
    for number, match in enumerate(display_pattern.finditer(math_text), start=1):
        validate_tex_expression(
            match.group(1),
            f"{path.relative_to(ROOT)} display equation {number}",
            errors,
        )
    inline_text = display_pattern.sub("", math_text)
    for line_number, line in enumerate(inline_text.splitlines(), start=1):
        if re.search(r"-\$(?!\$)", line):
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number} places inline math "
                "immediately after a hyphen, which GitHub may leave unrendered; "
                "rewrite the phrase with a space before $"
            )
        unescaped_dollars = re.findall(r"(?<!\\)\$", line)
        if len(unescaped_dollars) % 2:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number} has an unmatched inline $"
            )
            continue
        expressions = re.findall(r"(?<!\\)\$([^$\n]+?)(?<!\\)\$", line)
        for number, expression in enumerate(expressions, start=1):
            if expression.endswith("*"):
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_number} inline equation "
                    f"{number} ends in a bare *, which GitHub may parse as "
                    r"Markdown emphasis; use \ast inside braces"
                )
            validate_tex_expression(
                expression,
                f"{path.relative_to(ROOT)}:{line_number} inline equation {number}",
                errors,
            )

    prose_blocks = 0
    prose_text = re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.DOTALL)
    for block in re.split(r"\n\s*\n", prose_text):
        stripped = block.strip()
        if not stripped:
            continue
        if stripped.startswith(("#", "!", "|", "- ", "* ", ">", "$$")):
            continue
        if len(re.findall(r"[A-Za-z]+", stripped)) >= 25:
            prose_blocks += 1
    if prose_blocks < 15:
        errors.append(
            f"{path.relative_to(ROOT)} needs more sustained explanatory paragraphs"
        )


def validate_svg(path: Path, errors: list[str]) -> None:
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except (OSError, ET.ParseError) as exc:
        errors.append(f"{path.relative_to(ROOT)} is not valid SVG XML: {exc}")
        return

    namespace = {"svg": "http://www.w3.org/2000/svg"}
    if not root.get("viewBox"):
        errors.append(f"{path.relative_to(ROOT)} must declare a viewBox")
    title = root.find("svg:title", namespace)
    desc = root.find("svg:desc", namespace)
    if title is None or not "".join(title.itertext()).strip():
        errors.append(f"{path.relative_to(ROOT)} must include a descriptive title")
    if desc is None or not "".join(desc.itertext()).strip():
        errors.append(f"{path.relative_to(ROOT)} must include a descriptive description")
    if root.findall(".//svg:script", namespace):
        errors.append(f"{path.relative_to(ROOT)} must not contain scripts")


def markdown_text(path: Path) -> str:
    if path.suffix != ".ipynb":
        return path.read_text(encoding="utf-8")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    )


def validate_links(path: Path, errors: list[str]) -> None:
    text = markdown_text(path)
    targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    for target in targets:
        if target.startswith(("http://", "https://", "#")):
            continue
        target = target.split("#", 1)[0]
        if not target:
            continue
        resolved = (path.parent / target).resolve()
        if not resolved.is_file():
            errors.append(f"{path.relative_to(ROOT)} has missing local link {target}")


def main() -> int:
    errors: list[str] = []

    coverage = {
        "lecture": collect_basenames(LECTURES, "md"),
        "notebook": collect_basenames(IMPLEMENTATIONS, "ipynb"),
        "image": {
            basename
            for basename in EXPECTED_BASENAMES
            if (IMAGES / f"{basename}.svg").is_file()
        },
    }
    for kind, observed in coverage.items():
        missing = sorted(EXPECTED_BASENAMES - observed)
        extra = sorted(observed - EXPECTED_BASENAMES)
        if missing:
            errors.append(f"missing {kind} lessons: {', '.join(missing)}")
        if extra:
            errors.append(f"unexpected {kind} lessons: {', '.join(extra)}")

    for path in sorted(IMPLEMENTATIONS.glob("[0-9][0-9]_*.ipynb")):
        validate_notebook(path, errors)
        validate_links(path, errors)

    for path in sorted(LECTURES.glob("[0-9][0-9]_*.md")):
        validate_lecture(path, errors)
        validate_links(path, errors)

    validate_links(ROOT / "README.md", errors)

    for path in sorted(IMAGES.glob("[0-9][0-9]_*.svg")):
        validate_svg(path, errors)
        lesson_number = path.name[:2]
        matching_lectures = sorted(LECTURES.glob(f"{lesson_number}_*.md"))
        if len(matching_lectures) == 1:
            lecture_text = matching_lectures[0].read_text(encoding="utf-8")
            if path.name not in lecture_text:
                errors.append(
                    f"{path.relative_to(ROOT)} is not referenced by its lecture"
                )
    validate_svg(IMAGES / "curriculum_map.svg", errors)

    authored_files = [ROOT / "README.md"]
    authored_files.extend(LECTURES.glob("*.md"))
    authored_files.extend(IMPLEMENTATIONS.glob("*.ipynb"))
    authored_files.extend(IMAGES.glob("*.svg"))
    for path in authored_files:
        if path.is_file() and "\u2014" in path.read_text(encoding="utf-8"):
            errors.append(f"{path.relative_to(ROOT)} contains an em dash")

    if errors:
        print(f"Tutorial validation failed with {len(errors)} issue(s):")
        for error in errors:
            print(f"  - {error}")
        return 1

    svg_count = len(list(IMAGES.glob("*.svg")))
    print(
        "Tutorial validation passed: "
        f"{len(EXPECTED_BASENAMES)} lectures, {len(EXPECTED_BASENAMES)} notebooks, "
        f"and {svg_count} SVG figures."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
