import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Windows reserved device names that hang when accessed
WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
    "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9"
}

# Directories to skip
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", "coverage",
    ".angular", ".svelte-kit", "target", "bin", "obj",
    ".pytest_cache", ".mypy_cache", ".tox", "egg-info",
    ".bmad-core", ".claude", "web-bundles"
}

# Binary/non-code extensions to skip
SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".bmp", ".webp",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp3", ".mp4", ".wav", ".ogg", ".webm",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".pyc", ".pyo", ".class", ".o", ".obj",
    ".lock", ".sum",
}


def load_project_dir() -> Path:
    """Load the starting_project_directory from config.json."""
    config_path = Path(__file__).parent / "config.json"

    if not config_path.exists():
        print(f"Error: config.json not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    project_dir = Path(config["starting_project_directory"])

    if not project_dir.exists():
        print(f"Error: Project directory does not exist: {project_dir}")
        sys.exit(1)

    return project_dir


def count_lines(file_path: Path) -> int | None:
    """Count lines in a file. Returns None if file can't be read."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except (OSError, PermissionError):
        return None


def main():
    project_dir = load_project_dir()

    log_path = Path(__file__).parent / "count_lines.log"
    print(f"Counting lines in: {project_dir}")
    print(f"Directory log: {log_path}")
    print()

    by_extension = defaultdict(lambda: {"files": 0, "lines": 0})
    by_directory = defaultdict(lambda: {"files": 0, "lines": 0})
    total_files = 0
    total_lines = 0
    skipped_files = 0

    log_file = open(log_path, "w", encoding="utf-8")

    for dirpath, dirs, files in os.walk(project_dir):
        # Prune directories in-place to prevent descending into them
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS
            and not d.startswith(".")
            and d.lower().split(".")[0] not in WINDOWS_RESERVED
        ]

        rel_dir = Path(dirpath).relative_to(project_dir)
        dir_lines = 0
        dir_files = 0
        dir_file_entries = []

        for filename in files:
            # Skip reserved device names
            stem = filename.rsplit(".", 1)[0] if "." in filename else filename
            if stem.lower() in WINDOWS_RESERVED:
                continue

            file_path = Path(dirpath) / filename

            ext = file_path.suffix.lower()
            if ext in SKIP_EXTENSIONS:
                skipped_files += 1
                continue

            rel = file_path.relative_to(project_dir)
            print(f"  {rel}", end="", flush=True)

            lines = count_lines(file_path)
            if lines is None:
                print(" (unreadable)", flush=True)
                dir_file_entries.append(f"  {filename} (unreadable)\n")
                skipped_files += 1
                continue

            print(f"  ({lines:,} lines)", flush=True)
            dir_file_entries.append(f"  {filename} ({lines:,} lines)\n")

            total_files += 1
            total_lines += lines
            dir_lines += lines
            dir_files += 1

            by_extension[ext or "(no extension)"]["files"] += 1
            by_extension[ext or "(no extension)"]["lines"] += lines

            top_dir = rel.parts[0] if len(rel.parts) > 1 else "(root)"
            by_directory[top_dir]["files"] += 1
            by_directory[top_dir]["lines"] += lines

        if dir_file_entries:
            log_file.write(f"[DIR] {rel_dir} ({dir_files} files, {dir_lines:,} lines)\n")
            for entry in dir_file_entries:
                log_file.write(entry)
            log_file.write("\n")
            log_file.flush()

    log_file.close()

    # Print results
    print("=" * 60)
    print("BY FILE EXTENSION")
    print("=" * 60)
    print(f"{'Extension':<20} {'Files':>8} {'Lines':>10}")
    print("-" * 40)
    for ext, data in sorted(by_extension.items(), key=lambda x: x[1]["lines"], reverse=True):
        print(f"{ext:<20} {data['files']:>8} {data['lines']:>10}")

    print()
    print("=" * 60)
    print("BY TOP-LEVEL DIRECTORY")
    print("=" * 60)
    print(f"{'Directory':<30} {'Files':>8} {'Lines':>10}")
    print("-" * 50)
    for dir_name, data in sorted(by_directory.items(), key=lambda x: x[1]["lines"], reverse=True):
        print(f"{dir_name:<30} {data['files']:>8} {data['lines']:>10}")

    print()
    print("=" * 60)
    print("TOTALS")
    print("=" * 60)
    print(f"Total files counted:  {total_files}")
    print(f"Total lines:          {total_lines:,}")
    print(f"Skipped files:        {skipped_files}")
    print()


if __name__ == "__main__":
    main()
