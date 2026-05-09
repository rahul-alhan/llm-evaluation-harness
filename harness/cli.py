"""CLI front-end: register prompts, run eval, diff versions."""
from __future__ import annotations

import argparse
import json

from harness.runner import run as run_eval
from prompt_versioning.registry import diff, list_all, register


def _cmd_register(args):
    pv = register(name=args.name, file=args.file, author=args.author)
    print(f"Registered {pv.name} v{pv.version} ({pv.hash}) → {pv.body_path}")


def _cmd_evaluate(args):
    result = run_eval(
        prompt_name=args.prompt,
        eval_path=args.eval_set,
        out_path=args.out,
        version=args.version,
    )
    print(json.dumps(result["metrics"], indent=2, default=str))
    print(f"\nResult written to {args.out}")
    print(f"Gates passed: {result['gates_passed']}")


def _cmd_diff(args):
    print(diff(args.name, args.va, args.vb))


def _cmd_list(_args):
    for e in list_all():
        print(f"{e['name']:25s} v{e['version']:<3} {e['hash']:12s} {e['author']:12s} {e['created_at']}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register")
    r.add_argument("--name", required=True)
    r.add_argument("--file", required=True)
    r.add_argument("--author", required=True)
    r.set_defaults(func=_cmd_register)

    e = sub.add_parser("evaluate")
    e.add_argument("--prompt", required=True)
    e.add_argument("--version", type=int, default=None)
    e.add_argument("--eval-set", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(func=_cmd_evaluate)

    d = sub.add_parser("diff")
    d.add_argument("--name", required=True)
    d.add_argument("--va", type=int, required=True)
    d.add_argument("--vb", type=int, required=True)
    d.set_defaults(func=_cmd_diff)

    l = sub.add_parser("list")
    l.set_defaults(func=_cmd_list)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
