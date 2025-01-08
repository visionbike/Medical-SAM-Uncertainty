from types import NoneType
from typing import Any, List, Optional
from argparse import Namespace
from abc import ABC
from pathlib import Path, PosixPath
import torch
from yacs.config import CfgNode as cn
from .config_base import *

__all__ = [
    "ConfigYAML"
]


class ConfigYAML(ConfigBase, ABC):
    VALID_TYPES = {tuple, list, str, int, float, bool, NoneType, torch.Tensor}

    """
    The custom configuration parser for YAML config files.
    """

    def __init__(self, description: str) -> None:
        """
        Args:
            description (str): description for the parser.
        """
        super().__init__()
        self.parser.description = description
        # initial parser
        self.init_args()

    def _convert_to_dict(self, cfg_node: Any, keys: Optional[List] = None) -> Any:
        """
        Convert a configure node to dictionary.

        :param cfg_node: the input configuration node.
        :param keys: the list of keys.
        :return: a dictionary of CfgNode.
        """
        if not isinstance(cfg_node, cn):
            if type(cfg_node) not in self.VALID_TYPES:
                print(type(cfg_node))
                print(f"Key '{keys}' with value {cfg_node} is invalid.")
            return cfg_node
        else:
            if keys is None:
                keys = []
            cfg_dict = dict(cfg_node)
            for k, v in cfg_dict.items():
                cfg_dict[k] = self._convert_to_dict(v, keys + [k])
            return cfg_dict

    def _convert_to_namespace(self, cfg_node: Any, keys: Optional[List] = None) -> Any:
        if not isinstance(cfg_node, cn):
            if type(cfg_node) not in self.VALID_TYPES:
                print(f"Key '{keys}' with value {cfg_node} is invalid.")
            return cfg_node
        else:
            if keys is None:
                keys = []
            cfg_dict = dict(cfg_node)
            namespace = Namespace()
            for k, v in cfg_dict.items():
                setattr(namespace, k, self._convert_to_namespace(v, keys + [k]))
            return namespace

    def _add_arguments(self):
        self.parser.add_argument("-cfg", default="./cfgs/default_train.yaml", help="the path of YAML config file.")

    def _print_args(self):
        super()._print_args()

    def init_args(self):
        self._add_arguments()

    def parse(self, cmd_args: list = None):
        """
        :param cmd_args: command arguments.
        :return: the output ConfigNode.
        """
        self.args = self.parser.parse_args(cmd_args)

        print("### Load the YAML config file...")
        with open(self.args.cfg, "r") as f:
            cfgs = cn.load_cfg(f)
            print(f"Successfully loading the config YAML file!")

        cfgs.NetworkConfig.image_size = cfgs.DataConfig.image_size
        cfgs.NetworkConfig.multimask_output = cfgs.DataConfig.multimask_output
        if cfgs.ExpConfig.pretrain is not None:
            Path("pretrain_models").mkdir(parents=True, exist_ok=True)
            cfgs.ExpConfig.pretrain =f"pretrain_models/{cfgs.ExpConfig.pretrain}"
        cfgs.ExpConfig.exp_name = f"{cfgs.ExpConfig.mode}_{cfgs.DataConfig.dataset}_{cfgs.NetworkConfig.net}_{cfgs.NetworkConfig.block}_{cfgs.DataConfig.image_size}"
        cfgs = self._convert_to_namespace(cfgs)
        return cfgs
