import os
import os.path as osp
import platform
from pathlib import Path
from typing import Any, Generator, List, Optional, Union

from .misc import is_str

if platform.system() == "Windows":
    import regex as re
else:
    import re


def check_file_exist(filename: str, msg_tmpl: str = 'file "{}" does not exist') -> None:
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def mkdir_or_exist(dir_name: str, mode: int = 0o777) -> None:
    if dir_name == "":
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def symlink(src: str, dst: str, overwrite: bool = True, **kwargs) -> None:
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


def is_filepath(x: Any) -> bool:
    return is_str(x) or isinstance(x, Path)


def scandir(
    dir_path: Union[str, Path],
    suffix: Optional[str] = None,
    recursive: bool = False,
    case_sensitive: bool = True,
) -> Generator[str, None, None]:
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | :obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        case_sensitive (bool, optional) : If set to False, ignore the case of
            suffix. Default: True.
    Returns:
        A generator for all the interested files with relative paths.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    if suffix is not None and not case_sensitive:
        suffix = suffix.lower() if isinstance(suffix, str) else tuple(item.lower() for item in suffix)

    root = dir_path

    def _scandir(
        dir_path: Union[str, Path],
        suffix: Optional[str],
        recursive: bool,
        case_sensitive: bool,
    ) -> Generator[str, None, None]:
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                _rel_path = rel_path if case_sensitive else rel_path.lower()
                if suffix is None or _rel_path.endswith(suffix):
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(entry.path, suffix, recursive, case_sensitive)

    return _scandir(dir_path, suffix, recursive, case_sensitive)


def find_files(directory, pattern, recursive=True, abspath=False) -> List[str]:
    regex = re.compile(pattern)
    file_list = []
    for root, _, files in os.walk(directory):
        for f in files:
            if regex.match(f) is not None:
                file_list.append(os.path.join(root, f))
        if not recursive:
            break
    map_func = os.path.abspath if abspath else os.path.relpath
    return list(map(map_func, sorted(file_list)))


def natural_keys(text: str, retoken: str = r"[a-zA-Z]*(\d+)[a-zA-Z_]*[\.].*", n: int = 1) -> Union[int, str]:
    def _atoi(text: str) -> Union[int, str]:
        return int(text) if text.isdigit() else text.lower()

    return _atoi(re.split(retoken, text)[n])


listdirs = lambda root: [osp.join(base, d) for base, dirs, _ in os.walk(root) if dirs for d in dirs]

listfiles = lambda root: [f for base, _, files in os.walk(root) if files for f in files]


def parse_dirs_and_sort(
    input_dirs: Union[list, str],
    suffix: str,
    is_sort: bool = False,
    with_prefix: bool = True,
) -> List[str]:
    if isinstance(input_dirs, list):
        input_dirs_list = []
        for iter_input_dir in input_dirs:
            if osp.isdir(iter_input_dir):
                input_dirs_list += [
                    osp.join(iter_input_dir, x) if with_prefix else x
                    for x in scandir(
                        iter_input_dir,
                        suffix=suffix,
                        recursive=True,
                        case_sensitive=False,
                    )
                ]
            elif osp.isfile(iter_input_dir):
                if iter_input_dir.endswith(suffix):
                    input_dirs_list += [iter_input_dir]
            else:
                raise ValueError(f"Input path {iter_input_dir} is not exist.")
    elif isinstance(input_dirs, str):
        if osp.isdir(input_dirs):
            input_dirs_list = [
                osp.join(input_dirs, x) if with_prefix else x
                for x in scandir(input_dirs, suffix=suffix, recursive=True, case_sensitive=False)
            ]
        elif osp.isfile(input_dirs):
            if input_dirs.endswith(suffix):
                input_dirs_list = [input_dirs]
            else:
                input_dirs_list = []
        else:
            raise ValueError(f"Input path {input_dirs} is not exist.")
    else:
        raise ValueError("Only support list or str input.")

    if is_sort:
        try:
            try:
                input_dirs_list = sorted(
                    input_dirs_list,
                    key=lambda text: (
                        natural_keys(text, retoken=r"[a-zA-Z]*(\d+)_[0-9a-zA-Z_]*[\.].*", n=1),
                        natural_keys(text, retoken=r"[0-9a-zA-Z]*_(\d+)[a-zA-Z_]*[\.].*", n=1),
                    ),
                )
            except:
                input_dirs_list = sorted(input_dirs_list, key=lambda text: (natural_keys(text)))
        except:
            input_dirs_list = sorted(input_dirs_list, key=lambda text: text)

    return input_dirs_list
