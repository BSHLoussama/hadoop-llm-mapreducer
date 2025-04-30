#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

def get_connectors(use_ascii: bool):
    """
    Return a dict of tree connectors, either Unicode or pure ASCII.
    """
    if use_ascii:
        return {
            "branch": "+-- ",
            "last":   "`-- ",
            "pipe":   "|   ",
            "space":  "    ",
        }
    else:
        return {
            "branch": "├── ",
            "last":   "└── ",
            "pipe":   "│   ",
            "space":  "    ",
        }

def write_tree(dir_path, file_handle, prefix, conns):
    """
    Recursively write the directory tree of dir_path into file_handle,
    using the provided connectors.
    """
    try:
        entries = sorted(os.listdir(dir_path))
    except PermissionError:
        file_handle.write(f"{prefix}{conns['last']}[Permission Denied]\n")
        return

    for idx, name in enumerate(entries):
        full_path = os.path.join(dir_path, name)
        is_last = (idx == len(entries) - 1)
        connector = conns["last"] if is_last else conns["branch"]
        file_handle.write(f"{prefix}{connector}{name}\n")

        if os.path.isdir(full_path):
            extension = conns["space"] if is_last else conns["pipe"]
            write_tree(full_path, file_handle, prefix + extension, conns)

def main():
    parser = argparse.ArgumentParser(
        description="Generate a text file containing the directory-tree of a given folder."
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        default="project",
        help="Path to the folder to scan (default: ./project)"
    )
    parser.add_argument(
        "-o", "--output",
        default="architecture.txt",
        help="Output text file (default: architecture.txt)"
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Force pure ASCII connectors instead of Unicode (└──, ├──, etc.)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        print(f"Error: Folder '{args.input_folder}' does not exist.")
        sys.exit(1)

    # On Windows, default to ASCII unless overridden
    is_windows = sys.platform.startswith("win")
    use_ascii = args.ascii or is_windows

    connectors = get_connectors(use_ascii)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"{args.input_folder}\n")
        write_tree(args.input_folder, f, prefix="", conns=connectors)

    print(f"Directory tree written to '{args.output}'")

if __name__ == "__main__":
    main()