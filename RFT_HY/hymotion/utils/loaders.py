import importlib
import json
import os


def load_object(module_name, module_args, **extra_args):
    module_args = module_args.copy()
    module_path = ".".join(module_name.split(".")[:-1]).replace("/", ".")
    module = importlib.import_module(module_path)
    name = module_name.split(".")[-1]
    if module_args is None:
        module_args = {}
    module_args.update(extra_args)
    obj = getattr(module, name)(**module_args)
    return obj


def load_module(module_name):
    module_path = module_name.split(".")[0].replace("/", ".")
    module = importlib.import_module(module_path)
    name = module_name.split(".")[-1]
    obj = getattr(module, name)
    return obj


def check_cfg(cfg, global_dict, verbose=True):
    for key, val in cfg.items():
        if isinstance(val, dict):
            check_cfg(val, global_dict, verbose)
        elif isinstance(val, str):
            if val.startswith("$"):
                if verbose:
                    print(f" - Update {key} with {val} = {global_dict[val[1:]]}")
                cfg[key] = global_dict[val[1:]]


def read_yaml(yamlname):
    import yaml

    with open(yamlname, "r", encoding="utf-8") as file:
        try:
            data = yaml.safe_load(file)
        except yaml.constructor.ConstructorError:
            file.seek(0)
            data = yaml.load(file, Loader=yaml.FullLoader)
            if hasattr(data, "to_dict"):
                data = data.to_dict()
            elif hasattr(data, "_cfg_dict"):
                data = dict(data._cfg_dict)

    return data


def write_yaml(data, yamlname):
    import yaml

    with open(yamlname, "w", encoding="utf-8") as file:
        yaml.dump(data, file)


def check_input(data, verbose=True):
    data_parent = {}
    if "input" in data:
        if verbose:
            print(" - Check input file list")
        for filename in data.pop("input"):
            cfg_new = read_yaml(filename)
            data_parent.update(cfg_new)
    return data_parent


def merge_dict(dict_A, dict_B, key, verbose=True):
    if isinstance(dict_A[key], dict):
        dict_B = dict_B.copy()
        for key2, val2 in dict_A[key].items():
            if key2 in dict_B[key]:
                merge_dict(dict_A[key], dict_B[key], key2, verbose)
                dict_B[key].pop(key2)
        if len(dict_B[key]) > 0:
            if verbose:
                print(f" - Create {key} with {dict_B[key]}")
            for key2, val2 in dict_B[key].items():
                dict_A[key][key2] = val2
    else:
        if verbose:
            print(f" - Update {key} with {dict_B[key]}")
        dict_A[key] = dict_B[key]


def read_config(cfgname, verbose=True):
    data_base = read_yaml(cfgname)
    data_parent = check_input(data_base, verbose)
    # merge the data_base to data_parent
    for key, val in data_parent.items():
        if key in data_base:
            merge_dict(data_parent, data_base, key, verbose)
            if verbose:
                print(data_parent[key])
            data_base.pop(key)
    data_parent.update(data_base)
    data = data_parent
    check_cfg(data, data, verbose)
    return data


def update_config(config, args):
    for key, value in vars(args).items():
        if key in config.keys() and value is not None:
            config[key] = value


def read_yaml_full(path):
    import yaml

    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def check_ceph_path(path):
    import os

    if os.path.exists(path):
        return path
    else:
        raise ValueError(f"{path} not found")


def read_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_h5_dataset(filename, ds_name_list=None, parser=None):
    import h5py

    # ds for dataset
    if "@" in filename:
        filename, start_end = filename.split("@")
        start = int(start_end.split(":")[0])
        end = int(start_end.split(":")[1])
    else:
        start = None
        end = None
    assert os.path.isfile(filename), "cannot find: {}".format(filename)

    def load_dict(d):
        ds_dict = {}
        for item in d.keys():
            if ds_name_list is not None and item not in ds_name_list:
                continue
            if isinstance(d[item], h5py._hl.dataset.Dataset):
                ds_dict[item] = d[item][()]
                if parser is not None and item in parser:
                    ds_dict[item] = parser[item](ds_dict[item])
            elif isinstance(d[item], h5py._hl.group.Group):
                ds_dict[item] = load_dict(d[item])
        for item in d.attrs.keys():
            ds_dict[item] = d.attrs[item]
        return ds_dict

    with h5py.File(filename, "r") as f:
        ds_dict = load_dict(f)
        for item in f.attrs.keys():
            ds_dict[item] = f.attrs[item]
    if start is not None and end is not None:
        for key in ["LclRotation", "LclTranslation"]:
            ds_dict[key] = ds_dict[key][start:end]
    return ds_dict


if __name__ == "__main__":
    # hymotion.utils.loaders
    network = load_object("hymotion.utils.base_example.ToyNetwork", {})
    print(network)
    network = load_object("hymotion/utils/base_example.ToyNetwork", {})
    print(network)
    load_object("diffusers.DDIMScheduler", {})
    module = load_object("torch.nn.MSELoss", {"reduction": "none"})
    print(module)
