#!/usr/bin/env python3
"""
verify_precursor_equivalence.py

Decide mechanically whether two versions of a Python file differ only in
comments, docstrings and formatting, or whether executable behaviour changed.

Usage:
    python3 verify_precursor_equivalence.py REGISTERED.py CURRENT.py

Exit status:
    0  the two files are behaviourally equivalent (comments/docstrings only)
    1  executable content differs
    2  could not decide (parse error)

Method
------
Both files are parsed to an abstract syntax tree. Every docstring (module,
class, function) is removed, and all line/column attributes are discarded.
Comments never reach the AST at all. If the two normalized trees are
identical, the files execute the same bytecode-relevant structure, so no
behavioural change occurred. This is a stronger statement than "the diff
looks cosmetic".

Caveat stated plainly: AST equality does not cover a program that reads its
own __doc__ and branches on it. Neither of these files does, and the script
checks for that explicitly rather than assuming it.
"""

import ast
import difflib
import io
import sys
import tokenize


def strip_docstrings(tree):
    """Remove docstring statements from module, class and function bodies."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef,
                                 ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return tree


def normalized_dump(path):
    src = open(path, "rb").read()
    tree = ast.parse(src, filename=path)
    tree = strip_docstrings(tree)
    # include_attributes=False discards lineno/col_offset, so pure
    # reformatting and comment edits cannot show up as a difference.
    return ast.dump(tree, include_attributes=False, indent=1), src


def uses_own_docstring(path):
    """Flag the one case where a docstring edit could change behaviour."""
    src = open(path, "r", encoding="utf-8", errors="replace").read()
    hits = [ln for ln in src.splitlines()
            if "__doc__" in ln and not ln.strip().startswith("#")]
    return hits


def code_only_lines(path):
    """Source with comments stripped, for a human-readable diff."""
    out = []
    with open(path, "rb") as f:
        toks = list(tokenize.tokenize(f.readline))
    drop = {tokenize.COMMENT, tokenize.NL}
    prev_line = None
    for tok in toks:
        if tok.type in drop:
            continue
        if tok.start[0] != prev_line:
            prev_line = tok.start[0]
            line = tok.line.rstrip("\n")
            if line.strip():
                out.append(line)
    return out


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    reg, cur = argv[1], argv[2]

    try:
        dump_reg, src_reg = normalized_dump(reg)
        dump_cur, src_cur = normalized_dump(cur)
    except SyntaxError as e:
        print(f"PARSE ERROR: {e}")
        return 2

    print(f"registered : {reg}  ({len(src_reg)} bytes)")
    print(f"current    : {cur}  ({len(src_cur)} bytes)")
    print(f"byte-identical : {src_reg == src_cur}")

    for label, path in (("registered", reg), ("current", cur)):
        hits = uses_own_docstring(path)
        if hits:
            print(f"\nWARNING: {label} references __doc__, so a docstring "
                  f"edit could be behavioural. Inspect by hand:")
            for h in hits:
                print(f"    {h.strip()}")

    if dump_reg == dump_cur:
        print("\nAST VERDICT: EQUIVALENT")
        print("The two versions differ only in comments, docstrings or "
              "formatting.\nExecutable content is identical, so results "
              "produced under either\nversion are the same. Restoring the "
              "registered file is safe and\nrequires no re-run.")
        rc = 0
    else:
        print("\nAST VERDICT: EXECUTABLE CONTENT DIFFERS")
        print("The change is not cosmetic. Restore the registered file and "
              "re-run\nthe predictability sweep; report whichever numbers "
              "come out, with the\npost-registration modification disclosed "
              "in the ledger.")
        d = difflib.unified_diff(dump_reg.splitlines(), dump_cur.splitlines(),
                                 "registered.ast", "current.ast", lineterm="",
                                 n=2)
        print("\n--- normalized AST diff (first 60 lines) ---")
        for i, line in enumerate(d):
            if i >= 60:
                print("    ... truncated")
                break
            print(line)
        rc = 1

    print("\n--- source diff, comments stripped (first 80 lines) ---")
    d2 = difflib.unified_diff(code_only_lines(reg), code_only_lines(cur),
                              reg, cur, lineterm="", n=2)
    shown = 0
    for line in d2:
        if shown >= 80:
            print("    ... truncated")
            break
        print(line)
        shown += 1
    if shown == 0:
        print("(no differences outside comments)")

    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
