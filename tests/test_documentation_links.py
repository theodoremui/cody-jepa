import re
import unittest
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"\[[^]]*\]\(([^)]+)\)")


def maintained_markdown_files():
    yield PROJECT_ROOT / "README.md"
    for directory in ("notes", "reports", "tutorials"):
        yield from sorted((PROJECT_ROOT / directory).rglob("*.md"))


class DocumentationLinkTest(unittest.TestCase):
    def test_local_markdown_links_resolve(self):
        missing = []
        for document in maintained_markdown_files():
            for line_number, line in enumerate(document.read_text().splitlines(), 1):
                for raw_target in MARKDOWN_LINK.findall(line):
                    target = raw_target.split("#", 1)[0].strip()
                    if not target or urlparse(target).scheme or target.startswith("mailto:"):
                        continue
                    resolved = (document.parent / unquote(target)).resolve()
                    if not resolved.exists():
                        missing.append(
                            f"{document.relative_to(PROJECT_ROOT)}:{line_number}: "
                            f"{raw_target} -> {resolved}"
                        )
        self.assertEqual(missing, [], "broken local Markdown links:\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
