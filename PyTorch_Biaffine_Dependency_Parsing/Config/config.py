# -*- coding: utf-8 -*-
"""
文件说明：基于 ConfigParser 的配置对象
本文件相对原版的工程化改造：
    1. 删除 __init__ 末尾的 `config.write(open(config_file,'w'))` 副作用
       —— 旧版每次启动都会重写原 config.cfg（顺序/格式可能漂移），是隐蔽 bug
    2. 新增 override(pairs) 方法：支持命令行 `--override Section.key=value` 不复制 cfg 改值
    3. 新增 dump_effective(path) 方法：训练时把当前生效配置（基础 cfg + override 后）落盘
       到产物目录 effective_config.cfg，读者直接看产物就知道当时跑了什么超参
"""

from configparser import ConfigParser
import os
from typing import Iterable, Tuple


class myconf(ConfigParser):
    def __init__(self, defaults=None):
        ConfigParser.__init__(self, defaults=defaults)
        self.add_sec = "Additional"

    # 保持 key 大小写不变，避免 'pretrained_embed_file' 被强制小写
    def optionxform(self, optionstr):
        return optionstr


class Configurable(myconf):
    def __init__(self, config_file):
        super().__init__()
        self.test = None
        self.train = None

        config = myconf()
        config.read(config_file, encoding="utf8")
        self._config = config
        self.config_file = config_file

        print("Loaded config file successfully:", config_file)
        for section in config.sections():
            for k, v in config.items(section):
                print("  [{}] {} = {}".format(section, k, v))

        # save_direction 是产物根目录，启动前提前建好；其余字段 lazy 用到时再读
        if not os.path.isdir(self.save_direction):
            os.makedirs(self.save_direction, exist_ok=True)
        # 注意：不再把 config 写回原文件！原版的 `config.write(open(config_file,'w'))`
        # 会在每次启动时改写 cfg（哪怕用户没动），是隐蔽 bug。

    # ------------------------------------------------------------------
    # 命令行覆盖支持
    # ------------------------------------------------------------------
    def override(self, items: Iterable[str]) -> None:
        """
        用 ["Section.key=value", ...] 格式的字符串覆盖底层 ConfigParser。
        覆盖后 property 取值会自动反映新值。

        例：cfg.override(["Train.epochs=2", "Optimizer.adam=False", "Optimizer.sgd=True"])
        """
        for raw in items:
            if "=" not in raw or "." not in raw.split("=", 1)[0]:
                raise ValueError(
                    "override 项格式应为 Section.key=value，收到: {}".format(raw))
            keyfull, value = raw.split("=", 1)
            section, key = keyfull.split(".", 1)
            if not self._config.has_section(section):
                raise KeyError("配置中没有 section: {}".format(section))
            if not self._config.has_option(section, key):
                # 允许新增字段，但打印警告，避免拼写错误悄无声息
                print("[WARN] 创建新字段: [{}] {} = {}".format(section, key, value))
            self._config.set(section, key, str(value))
            print("[override] [{}] {} -> {}".format(section, key, value))

    def dump_effective(self, path: str) -> str:
        """
        把当前生效的配置（基础 cfg + 命令行 override 后）写到 path。
        训练时调用，让产物目录 effective_config.cfg 与本次训练完全对齐。
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            self._config.write(f)
        return path

    # ------------------------------------------------------------------
    # 下面全部是只读 property，直接照搬旧版
    # ------------------------------------------------------------------
    # Embed
    @property
    def pretrained_embed(self):
        return self._config.getboolean('Embed', 'pretrained_embed')

    @property
    def zeros(self):
        return self._config.getboolean('Embed', 'zeros')

    @property
    def avg(self):
        return self._config.getboolean('Embed', 'avg')

    @property
    def uniform(self):
        return self._config.getboolean('Embed', 'uniform')

    @property
    def nnembed(self):
        return self._config.getboolean('Embed', 'nnembed')

    @property
    def pretrained_embed_file(self):
        return self._config.get('Embed', 'pretrained_embed_file')

    # Data
    @property
    def train_file(self):
        return self._config.get('Data', 'train_file')

    @property
    def dev_file(self):
        return self._config.get('Data', 'dev_file')

    @property
    def test_file(self):
        return self._config.get('Data', 'test_file')

    @property
    def max_count(self):
        return self._config.getint('Data', 'max_count')

    @property
    def min_freq(self):
        return self._config.getint('Data', 'min_freq')

    @property
    def shuffle(self):
        return self._config.getboolean('Data', 'shuffle')

    @property
    def epochs_shuffle(self):
        return self._config.getboolean('Data', 'epochs_shuffle')

    # Save
    @property
    def save_pkl(self):
        return self._config.getboolean('Save', 'save_pkl')

    @property
    def pkl_directory(self):
        return self._config.get('Save', 'pkl_directory')

    @property
    def pkl_data(self):
        return self._config.get('Save', 'pkl_data')

    @property
    def pkl_alphabet(self):
        return self._config.get('Save', 'pkl_alphabet')

    @property
    def pkl_iter(self):
        return self._config.get('Save', 'pkl_iter')

    @property
    def pkl_embed(self):
        return self._config.get('Save', 'pkl_embed')

    @property
    def save_dict(self):
        return self._config.getboolean('Save', 'save_dict')

    @property
    def save_direction(self):
        return self._config.get('Save', 'save_direction')

    @property
    def dict_directory(self):
        return self._config.get('Save', 'dict_directory')

    @property
    def word_dict(self):
        return self._config.get('Save', 'word_dict')

    @property
    def label_dict(self):
        return self._config.get('Save', 'label_dict')

    @property
    def model_name(self):
        return self._config.get('Save', 'model_name')

    @property
    def save_best_model_dir(self):
        return self._config.get('Save', 'save_best_model_dir')

    @property
    def save_model(self):
        return self._config.getboolean('Save', 'save_model')

    @property
    def save_all_model(self):
        return self._config.getboolean('Save', 'save_all_model')

    @property
    def save_best_model(self):
        return self._config.getboolean('Save', 'save_best_model')

    @property
    def rm_model(self):
        return self._config.getboolean('Save', 'rm_model')

    # Model
    @property
    def embed_dim(self):
        return self._config.getint("Model", "embed_dim")

    @property
    def tag_dims(self):
        return self._config.getint("Model", "tag_dims")

    @property
    def dropout_emb(self):
        return self._config.getfloat("Model", "dropout_emb")

    @property
    def dropout(self):
        return self._config.getfloat("Model", "dropout")

    @property
    def lstm_layers(self):
        return self._config.getint("Model", "lstm_layers")

    @property
    def lstm_hiddens(self):
        return self._config.getint("Model", "lstm_hiddens")

    @property
    def dropout_lstm_input(self):
        return self._config.getfloat("Model", "dropout_lstm_input")

    @property
    def dropout_lstm_hidden(self):
        return self._config.getfloat("Model", "dropout_lstm_hidden")

    @property
    def mlp_arc_size(self):
        return self._config.getint("Model", "mlp_arc_size")

    @property
    def mlp_rel_size(self):
        return self._config.getint("Model", "mlp_rel_size")

    @property
    def dropout_mlp(self):
        return self._config.getfloat("Model", "dropout_mlp")

    # Encoder (Stage 2 新增)
    # type=lstm 走 Stage 1 原版 BiLSTM；type=transformer 走 TransformerEncoder.py
    @property
    def encoder_type(self) -> str:
        # 若用户的 cfg 没有 [Encoder] 段（兼容旧 cfg），默认回退到 lstm
        if not self._config.has_section("Encoder"):
            return "lstm"
        return self._config.get("Encoder", "type").lower()

    @property
    def encoder_d_model(self) -> int:
        return self._config.getint("Encoder", "d_model")

    @property
    def encoder_nhead(self) -> int:
        return self._config.getint("Encoder", "nhead")

    @property
    def encoder_num_layers(self) -> int:
        return self._config.getint("Encoder", "num_layers")

    @property
    def encoder_ff_size(self) -> int:
        return self._config.getint("Encoder", "ff_size")

    @property
    def encoder_dropout(self) -> float:
        return self._config.getfloat("Encoder", "dropout")

    @property
    def encoder_norm_first(self) -> bool:
        return self._config.getboolean("Encoder", "norm_first")

    @property
    def encoder_activation(self) -> str:
        return self._config.get("Encoder", "activation").lower()

    # Optimizer
    @property
    def adam(self):
        return self._config.getboolean("Optimizer", "adam")

    @property
    def sgd(self):
        return self._config.getboolean("Optimizer", "sgd")

    @property
    def muon(self) -> bool:
        # 若用户 cfg 没有 muon 字段（兼容旧 cfg），默认 False
        if not self._config.has_option("Optimizer", "muon"):
            return False
        return self._config.getboolean("Optimizer", "muon")

    @property
    def learning_rate(self):
        return self._config.getfloat("Optimizer", "learning_rate")

    @property
    def weight_decay(self):
        return self._config.getfloat("Optimizer", "weight_decay")

    @property
    def clip_max_norm_use(self):
        return self._config.getboolean("Optimizer", "clip_max_norm_use")

    @property
    def clip_max_norm(self):
        return self._config.get("Optimizer", "clip_max_norm")

    @property
    def use_lr_decay(self):
        return self._config.getboolean("Optimizer", "use_lr_decay")

    @property
    def lr_rate_decay(self):
        return self._config.getfloat("Optimizer", "lr_rate_decay")

    @property
    def min_lrate(self):
        return self._config.getfloat("Optimizer", "min_lrate")

    @property
    def max_patience(self):
        return self._config.getint("Optimizer", "max_patience")

    # Train
    @property
    def num_threads(self):
        return self._config.getint("Train", "num_threads")

    @property
    def epochs(self):
        return self._config.getint("Train", "epochs")

    @property
    def early_max_patience(self):
        return self._config.getint("Train", "early_max_patience")

    @property
    def update_batch_size(self):
        return self._config.getint("Train", "update_batch_size")

    @property
    def batch_size(self):
        return self._config.getint("Train", "batch_size")

    @property
    def dev_batch_size(self):
        return self._config.getint("Train", "dev_batch_size")

    @property
    def test_batch_size(self):
        return self._config.getint("Train", "test_batch_size")

    @property
    def log_interval(self):
        return self._config.getint("Train", "log_interval")
