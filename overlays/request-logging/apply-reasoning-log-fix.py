#!/usr/bin/env python3
"""Apply the reasoning request-logging fix to an installed vLLM.

Content-based, not path-based: the two target files moved between releases
(0.21 / 0.23 / 0.27 / jasl / upstream all differ), but the code being fixed is
byte-identical in every one. Locates by content, writes .orig backups, and is
idempotent.

    python3 apply-reasoning-log-fix.py [--vllm-root DIR] [--check] [--revert]

Exit 0 = applied or already applied. Exit 1 = a target was not found.
"""
import argparse, pathlib, shutil, sys

NONSTREAM_OLD = """                if output_text:
                    # Get the corresponding output token IDs
                    output_token_ids = None"""
NONSTREAM_NEW = """                # Reasoning models route thinking to the reasoning field, so a
                # reasoning-only or truncated-mid-thought response leaves
                # content empty and would otherwise log nothing at all.
                reasoning_text = (
                    getattr(choice.message, "reasoning", None)
                    or getattr(choice.message, "reasoning_content", None)
                )
                if reasoning_text:
                    reasoning_part = f"[reasoning: {reasoning_text}]"
                    output_text = (
                        f"{reasoning_part} {output_text}"
                        if output_text
                        else reasoning_part
                    )

                if output_text:
                    # Get the corresponding output token IDs
                    output_token_ids = None"""

STREAM_OLD = """            if chat_response.choices and chat_response.choices[0].delta.content:
                return chat_response.choices[0].delta.content"""
STREAM_NEW = """            if chat_response.choices:
                delta = chat_response.choices[0].delta
                # Reasoning models stream thinking in delta.reasoning; without
                # it a reasoning-only response logs "no_content".
                return delta.content or getattr(delta, "reasoning", None) or \"\""""

# Each edit needs its own "already applied" marker, and the marker must be a
# string that exists ONLY after patching. Note NONSTREAM_NEW ends with
# NONSTREAM_OLD, so the old anchor still matches after a successful patch --
# checking the marker first is what makes this idempotent rather than
# double-applying.
EDITS = [
    ("non-streaming log_outputs", NONSTREAM_OLD, NONSTREAM_NEW,
     'reasoning_part = f"[reasoning: {reasoning_text}]"'),
    ("streaming response middleware", STREAM_OLD, STREAM_NEW,
     'return delta.content or getattr(delta, "reasoning", None)'),
]


def find_vllm_root(explicit):
    if explicit:
        return pathlib.Path(explicit)
    try:
        import vllm
        return pathlib.Path(vllm.__file__).parent
    except Exception:
        sys.exit("could not import vllm; pass --vllm-root")


def locate(root, needle):
    hits = [p for p in root.rglob("*.py") if needle in p.read_text(errors="ignore")]
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-root")
    ap.add_argument("--check", action="store_true", help="report only, change nothing")
    ap.add_argument("--revert", action="store_true", help="restore .orig backups")
    a = ap.parse_args()
    root = find_vllm_root(a.vllm_root)
    print(f"vllm root: {root}")

    if a.revert:
        n = 0
        for orig in root.rglob("*.py.orig"):
            shutil.copy2(orig, orig.with_suffix(""))
            orig.unlink(); n += 1
            print(f"  reverted {orig.with_suffix('').relative_to(root)}")
        print(f"reverted {n} file(s)")
        return 0

    rc = 0
    for label, old, new, marker in EDITS:
        already = locate(root, marker)
        if already:
            print(f"  [already applied] {label}: {already[0].relative_to(root)}")
            continue
        hits = locate(root, old)
        if not hits:
            print(f"  [NOT FOUND]      {label}  <-- investigate before deploying")
            rc = 1
            continue
        if len(hits) > 1:
            print(f"  [ambiguous]      {label}: {len(hits)} matches, refusing")
            rc = 1
            continue
        target = hits[0]
        print(f"  [{'would patch' if a.check else 'patched'}]     {label}: {target.relative_to(root)}")
        if not a.check:
            backup = target.with_suffix(target.suffix + ".orig")
            if not backup.exists():
                shutil.copy2(target, backup)
            target.write_text(target.read_text().replace(old, new, 1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
