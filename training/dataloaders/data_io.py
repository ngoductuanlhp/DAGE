import os
import shutil


def download_file(path):
    """Read a file from local filesystem and return its contents as bytes.

    Args:
        path: Local filesystem path to read.

    Returns:
        File contents as bytes.
    """
    with open(path, "rb") as f:
        return f.read()


def download_file_with_cache(path, cache_dir=None):
    """Read a file from local filesystem with optional caching.

    For local storage this is identical to download_file since the data
    is already on disk.

    Args:
        path: Local filesystem path to read.
        cache_dir: Unused. Kept for API compatibility.

    Returns:
        File contents as bytes.
    """
    return download_file(path)


def upload_file(local_path, remote_path):
    """Copy a local file to the target path.

    Args:
        local_path: Source file path.
        remote_path: Destination file path.
    """
    os.makedirs(os.path.dirname(remote_path), exist_ok=True)
    if os.path.abspath(local_path) != os.path.abspath(remote_path):
        shutil.copy2(local_path, remote_path)


def list_files(prefix):
    """List all files under a local directory prefix.

    Args:
        prefix: Local directory path to list.

    Returns:
        List of file paths (relative structure preserved).
    """
    results = []
    for root, _dirs, files in os.walk(prefix):
        for fname in files:
            results.append(os.path.join(root, fname))
    return sorted(results)


def check_path_exists(path):
    """Check whether a local path exists.

    Args:
        path: Local filesystem path to check.

    Returns:
        True if the path exists, False otherwise.
    """
    return os.path.exists(path)
