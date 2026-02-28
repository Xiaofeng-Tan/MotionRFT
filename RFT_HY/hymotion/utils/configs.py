import ast
import copy
import os.path as osp
import platform
import re
import shutil
import sys
import tempfile
import types
import uuid
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterator, NoReturn, Optional, Union
import yaml

from .misc import import_modules_from_strings
from .path import check_file_exist

BASE_KEY = "_base_"
DELETE_KEY = "_delete_"
RESERVED_KEYS = ["filename", "text", "pretty_text"]


class Config:
    def __init__(
        self,
        cfg_dict: Optional[dict] = None,
        cfg_text: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> None:
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError("cfg_dict must be a dict, but " f"got {type(cfg_dict)}")
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f"{key} is reserved for config file")

        if isinstance(filename, Path):
            filename = str(filename)

        super(Config, self).__setattr__("_cfg_dict", ConfigDict(cfg_dict))
        super(Config, self).__setattr__("_filename", filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, "r") as f:
                text = f.read()
        else:
            text = ""
        super(Config, self).__setattr__("_text", text)

    @staticmethod
    def fromfile(
        filename: str,
        use_predefined_variables: bool = True,
        import_custom_modules: bool = True,
    ) -> "Config":
        if isinstance(filename, Path):
            filename = str(filename)
        cfg_dict, cfg_text = Config._file2dict(filename, use_predefined_variables)
        if import_custom_modules and cfg_dict.get("custom_imports", None):
            import_modules_from_strings(**cfg_dict["custom_imports"])
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def _file2dict(filename: str, use_predefined_variables: bool = True) -> tuple[dict, str]:
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in [".py"]:
            raise IOError("Only py type are supported now!")

        cfg_dict = {}

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(dir=temp_config_dir, suffix=fileExtname)
            if platform.system() == "Windows":
                temp_config_file.close()
            temp_config_name = osp.basename(temp_config_file.name)
            # Substitute predefined variables
            if use_predefined_variables:
                Config._substitute_predefined_vars(filename, temp_config_file.name)
            else:
                shutil.copyfile(filename, temp_config_file.name)
            # Substitute base variables from placeholders to strings
            base_var_dict = Config._pre_substitute_base_vars(temp_config_file.name, temp_config_file.name)

            if filename.endswith(".py"):
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                Config._validate_py_syntax(filename)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith("__")
                    and not isinstance(value, types.ModuleType)
                    and not isinstance(value, types.FunctionType)
                }
                # delete imported module
                del sys.modules[temp_module_name]

            # close temp file
            temp_config_file.close()

        cfg_text = filename + "\n"
        with open(filename, "r", encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = base_filename if isinstance(base_filename, list) else [base_filename]

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                duplicate_keys = base_cfg_dict.keys() & c.keys()
                if len(duplicate_keys) > 0:
                    raise KeyError("Duplicate key is not allowed among bases. " f"Duplicate keys: {duplicate_keys}")
                base_cfg_dict.update(c)

            # Substitute base variables from strings to their actual values
            cfg_dict = Config._substitute_base_vars(cfg_dict, base_var_dict, base_cfg_dict)
            assert isinstance(cfg_dict, dict)

            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = "\n".join(cfg_text_list)

        return cfg_dict, cfg_text

    @staticmethod
    def _validate_py_syntax(filename: str) -> None:
        with open(filename, "r", encoding="utf-8") as f:
            # Setting encoding explicitly to resolve coding issue on windows
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError("There are syntax errors in config " f"file {filename}: {e}")

    @staticmethod
    def _pre_substitute_base_vars(filename: str, temp_config_name: str) -> dict:
        """Substitute base variable placehoders to string, so that parsing would work."""
        with open(filename, "r", encoding="utf-8") as f:
            config_file = f.read()
        base_var_dict = {}
        regexp = r"\{\{\s*" + BASE_KEY + r"\.([\w\.]+)\s*\}\}"
        base_vars = set(re.findall(regexp, config_file))
        for base_var in base_vars:
            randstr = f"_{base_var}_{uuid.uuid4().hex.lower()[:6]}"
            base_var_dict[randstr] = base_var
            regexp = r"\{\{\s*" + BASE_KEY + r"\." + base_var + r"\s*\}\}"
            config_file = re.sub(regexp, f'"{randstr}"', config_file)
        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)
        return base_var_dict

    @staticmethod
    def _substitute_base_vars(
        cfg: Union[dict, list, tuple, str],
        base_var_dict: dict,
        base_cfg: dict,
    ) -> Union[dict, list, tuple, str]:
        """Substitute variable strings to their actual values."""
        cfg = copy.deepcopy(cfg)

        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if isinstance(v, str) and v in base_var_dict:
                    new_v = base_cfg
                    for new_k in base_var_dict[v].split("."):
                        new_v = new_v[new_k]
                    cfg[k] = new_v
                elif isinstance(v, (list, tuple, dict)):
                    cfg[k] = Config._substitute_base_vars(v, base_var_dict, base_cfg)
        elif isinstance(cfg, tuple):
            cfg = tuple(Config._substitute_base_vars(c, base_var_dict, base_cfg) for c in cfg)
        elif isinstance(cfg, list):
            cfg = [Config._substitute_base_vars(c, base_var_dict, base_cfg) for c in cfg]
        elif isinstance(cfg, str) and cfg in base_var_dict:
            new_v = base_cfg
            for new_k in base_var_dict[cfg].split("."):
                new_v = new_v[new_k]
            cfg = new_v

        return cfg

    @staticmethod
    def _substitute_predefined_vars(filename: str, temp_config_name: str) -> None:
        file_dirname = osp.dirname(filename)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname,
        )
        with open(filename, "r", encoding="utf-8") as f:
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r"\{\{\s*" + str(key) + r"\s*\}\}"
            value = value.replace("\\", "/")
            config_file = re.sub(regexp, value, config_file)
        with open(temp_config_name, "w", encoding="utf-8") as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _merge_a_into_b(a: dict, b: dict, allow_list_keys: bool = False) -> dict:
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f"Index {k} exceeds the length of list {b}")
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            elif isinstance(v, dict):
                if k in b and not v.pop(DELETE_KEY, False):
                    allowed_types = (dict, list) if allow_list_keys else dict
                    if not isinstance(b[k], allowed_types):
                        raise TypeError(
                            f"{k}={v} in child config cannot inherit from "
                            f"base because {k} is a dict in the child config "
                            f"but is of type {type(b[k])} in base config. "
                            f"You may set `{DELETE_KEY}=True` to ignore the "
                            f"base config."
                        )
                    b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
                else:
                    b[k] = ConfigDict(v)
            else:
                b[k] = v
        return b

    def to_dict(self) -> Any:
        def convert_configdict(obj):
            if isinstance(obj, ConfigDict):
                return {k: convert_configdict(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: convert_configdict(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_configdict(item) for item in obj]
            else:
                return obj

        return convert_configdict(self._cfg_dict)

    @classmethod
    def from_dict(cls, cfg_dict: dict, filename: Optional[str] = None) -> "Config":
        return cls(cfg_dict=cfg_dict, filename=filename)

    def save_yaml(self, filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, indent=2)

    @classmethod
    def load_yaml(cls, filename: str) -> "Config":
        with open(filename, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        return cls.from_dict(cfg_dict, filename=filename)

    def __repr__(self) -> str:
        return f"Config (path: {self.filename}): {self._cfg_dict.__repr__()}"

    def __len__(self) -> int:
        return len(self._cfg_dict)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name: str) -> Any:
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name: str, value: Any) -> None:
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._cfg_dict)

    def __getstate__(self) -> tuple[dict, str, str]:
        return (self._cfg_dict, self._filename, self._text)

    def __copy__(self) -> "Config":
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)

        return other

    def __deepcopy__(self, memo: dict) -> "Config":
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            super(Config, other).__setattr__(key, copy.deepcopy(value, memo))

        return other


class ConfigDict(Dict):
    def __missing__(self, name: str) -> NoReturn:
        raise KeyError(name)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def to_dict(self) -> Any:
        def convert_configdict(obj):
            if isinstance(obj, ConfigDict):
                return {k: convert_configdict(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: convert_configdict(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_configdict(item) for item in obj]
            else:
                return obj

        return convert_configdict(dict(self))
