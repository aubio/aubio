"""Check a pull request title against Conventional Commits and suggest a fix.

main squash-merges with the PR title as the commit subject, and release-please
reads those subjects to choose the next version and write the changelog. A
title it cannot parse leaves the change out of the release notes.

Usage: TITLE="..." python3 pr_title.py COMMENT_FILE
Exits 0 when the title is valid. Otherwise writes the bot comment (Markdown)
to COMMENT_FILE and exits 1.
"""

import os
import re
import sys

MARKER = "<!-- ledfx-pr-title -->"

# type -> (what it is for, what it does to the release)
TYPES = {
    "feat": ("a new feature", "minor bump, listed under Features"),
    "fix": ("a bug fix", "patch bump, listed under Bug Fixes"),
    "perf": ("a performance improvement", "patch bump, listed"),
    "revert": ("reverting an earlier commit", "patch bump, listed"),
    "refactor": ("code changes that neither fix nor add", "not listed"),
    "docs": ("documentation only", "not listed"),
    "test": ("tests only", "not listed"),
    "build": ("packaging, installers, dependencies' build", "not listed"),
    "ci": ("GitHub Actions and other CI config", "not listed"),
    "chore": ("maintenance, tooling, dependency bumps", "not listed"),
    "style": ("formatting only", "not listed"),
}

VALID = re.compile(rf"^(?:{'|'.join(TYPES)})(?:\([^()\s]+\))?!?: \S")

# A type-like prefix written the wrong way: "Fix: x", "[Feat] x", "fix - x",
# "Feature(dmx): x", "fix:x", "fix noqa: x" (the second word becomes the scope).
PREFIX = re.compile(
    r"^\[?(?P<type>[A-Za-z]+)\]?(?:\((?P<scope>[^()]*)\)|\s+(?P<word>[\w.-]+)(?=\s*:))?(?P<bang>!)?"
    r"(?:\s*:\s*|\]\s+|\s+[-–—]\s+)(?P<rest>\S.*)$"
)

ALIASES = {
    **{t: t for t in TYPES},
    "feature": "feat",
    "features": "feat",
    "bugfix": "fix",
    "bug": "fix",
    "hotfix": "fix",
    "fixes": "fix",
    "fixed": "fix",
    "doc": "docs",
    "documentation": "docs",
    "tests": "test",
    "refactoring": "refactor",
    "performance": "perf",
    "deps": "chore(deps)",
    "dependencies": "chore(deps)",
}

# First word of an unprefixed title -> likely type.
VERBS = {
    "feat": [
        "add",
        "adds",
        "added",
        "implement",
        "implements",
        "introduce",
        "support",
        "new",
        "create",
        "enable",
        "allow",
    ],
    "fix": [
        "fix",
        "fixes",
        "fixed",
        "resolve",
        "resolves",
        "correct",
        "prevent",
        "handle",
        "repair",
        "avoid",
        "stop",
    ],
    "refactor": [
        "refactor",
        "rename",
        "move",
        "simplify",
        "clean",
        "cleanup",
        "restructure",
        "extract",
        "split",
        "remove",
        "tidy",
    ],
    "docs": ["document", "docs", "readme"],
    "chore(deps)": ["bump", "upgrade"],
    "perf": ["speed", "optimize", "optimise"],
    "test": ["test", "tests"],
    "revert": ["revert"],
}
VERB_TYPE = {verb: t for t, verbs in VERBS.items() for verb in verbs}
FIX_VERBS = {"fix", "fixes", "fixed"}  # "Fix crash" reads better as "fix: crash"


def _lower_first(text: str) -> str:
    """Lowercase a leading ordinary word; keep acronyms and names like 'LIFX'."""
    word = text.split(" ", 1)[0]
    if word[:1].isupper() and word[1:] == word[1:].lower():
        return text[0].lower() + text[1:]
    return text


def suggest(title: str) -> str:
    """Best-effort Conventional Commits version of an invalid title."""
    title = re.sub(r"^(?:wip|draft)\b[:\s]*", "", title.strip(), flags=re.IGNORECASE)
    title = title.rstrip(" .")

    m = PREFIX.match(title)
    if m and m["type"].lower() in ALIASES:
        kind = ALIASES[m["type"].lower()]
        scope = (m["scope"] or m["word"] or "").lower().replace(" ", "")
        if scope and not kind.endswith(")"):  # chore(deps) already has one
            kind = f"{kind}({scope})"
        rest = m["rest"]
        if rest.split(" ", 1)[0].lower() in VERB_TYPE:
            rest = _lower_first(rest)
        return f"{kind}{m['bang'] or ''}: {rest}"

    first, _, rest = title.partition(" ")
    kind = VERB_TYPE.get(first.lower(), "feat")
    if first.lower() in FIX_VERBS and rest:
        return f"fix: {rest}"
    return f"{kind}: {_lower_first(title)}"


def _fenced(text: str) -> str:
    """A code block the text cannot close, so a title cannot inject Markdown."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def comment(title: str) -> str:
    rows = "\n".join(
        f"| `{t}` | {use} | {effect} |" for t, (use, effect) in TYPES.items()
    )
    return f"""{MARKER}
👋 Thanks for the pull request! One thing before it can merge: **the title needs to follow [Conventional Commits](https://www.conventionalcommits.org/)**.

aubio-ledfx squash-merges every PR with its title as the commit message, and [release-please](https://github.com/googleapis/release-please) builds each release's version number and changelog from those messages. It cannot read this title, so the change would be left out of the release notes.

**Current title**
{_fenced(title)}

**Suggested title** (a best guess: change the type or add a scope if it isn't quite right)
{_fenced(suggest(title))}

The format is `type(optional-scope): summary`, with a lowercase type from this list:

| Type | Use for | In the release |
|---|---|---|
{rows}

A scope names the area, e.g. `fix(api): …`, `feat(effects): …`, `chore(deps): …`. Add `!` before the colon for a breaking change, e.g. `feat(api)!: …`.

Edit the title and this check runs again by itself; this comment is removed once it passes."""


def main() -> int:
    title = " ".join(os.environ.get("TITLE", "").split())
    if VALID.match(title):
        print(f"OK: {title}")
        return 0
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        f.write(comment(title))
    print(
        f"::error title=PR title is not a Conventional Commit::Suggested: {suggest(title)}"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
