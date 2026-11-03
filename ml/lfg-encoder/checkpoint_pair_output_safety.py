"""Output path checks for private checkpoint-pair evaluation reports."""
from __future__ import annotations

import os
from pathlib import Path
import stat


def _is_windows() -> bool:
    return os.name == 'nt'


def require_private_output_directory(output: Path) -> None:
    """Require a protected POSIX parent chain; Windows ACL creation is not implemented."""
    if _is_windows():
        raise ValueError('private output ACL verification is unavailable on Windows')
    absolute = Path(os.path.abspath(output))
    if absolute.exists() or absolute.is_symlink():
        raise ValueError('output already exists')
    parent = absolute.parent
    chain = []
    while True:
        chain.append(parent)
        if parent.parent == parent:
            break
        parent = parent.parent
    uid = os.getuid()
    for index, directory in enumerate(reversed(chain)):
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError('output path contains a non-directory component')
        if index == len(chain) - 1:
            if info.st_uid != uid or (info.st_mode & 0o077):
                raise ValueError('output parent must be owned and private to this account')
        elif (info.st_uid not in (0, uid)
              or (info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX)):
            raise ValueError('output path has an unsafe writable ancestor')


def create_private_output(output: Path) -> int:
    """Recheck the path, then exclusively create a private report file."""
    require_private_output_directory(output)
    return os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
