# -*- coding: utf-8 -*-
"""
文件说明：训练入口的辅助函数集合
本文件相对原版的工程化改造：
    1. 删除 `from test import load_test_model` —— 旧 test.py 是另一个任务（CAIL）
       的死代码，本项目不存在；保留 load_test_model 占位函数避免 import 链断裂
    2. save_dictionary 不再用 shutil.copytree 复制整个 dict 目录到 save_dir，
       直接把 word/ext_word 字典写到 save_dir 子目录下
    3. 把 print 替换为可选 logger（向后兼容：未传 logger 时仍 print）
"""

import os
import shutil
import time
import random

import torch

from DataUtils.Alphabet import CreateAlphabet
from DataUtils.Batch_Iterator import Iterators
from DataUtils.Embed import Embed
from DataUtils.Common import seed_num, cpu_device, PAD, print_common
from Dataloader.DataLoader import DataLoader
from Model.Biaffine_Parsing.Model import ParserModel
from Model.Biaffine_Parsing.Parser import BiaffineParser

torch.manual_seed(seed_num)
random.seed(seed_num)


def load_test_model(model, config):
    """
    占位：原版 test.py 不存在，独立推理还没做。
    若用户在 --test 模式下进入 load_model，会落到这里抛出明确报错而不是 ImportError。
    """
    raise NotImplementedError(
        "独立推理脚本未实现；要从 ckpt 恢复，请用 DataUtils.Checkpoint.load_checkpoint。"
    )


# ---------------------------------------------------------------------------
# 数据预处理：读 conll → 建词表 → 重新读以注入 ROOT → 组装 batch iterator
# ---------------------------------------------------------------------------
def preprocessing(config):
    """
    返回 (train_iter, dev_iter, test_iter, alphabet)。
    旧版有 save_pkl 分支把中间产物写盘，本版默认关闭以避免动辄几百 MB 的 pkl 文件。
    """
    print("Processing Data......")
    # 第一遍：用于统计词频
    data_loader = DataLoader(path=[config.train_file, config.dev_file, config.test_file],
                             shuffle=True, config=config)
    train_data, dev_data, test_data = data_loader.dataLoader()
    print("train sentence {}, dev sentence {}, test sentence {}.".format(
        len(train_data), len(dev_data), len(test_data)))

    # 建立四个词表（word / ext_word / tag / rel）
    alphabet = CreateAlphabet(min_freq=config.min_freq, train_data=train_data,
                              dev_data=dev_data, test_data=test_data, config=config)
    alphabet.build_vocab()

    # 第二遍：带 alphabet 重新读，会在每句首插 ROOT
    data_loader_alphabet = DataLoader(path=[config.train_file, config.dev_file, config.test_file],
                                      shuffle=True, config=config, alphabet=alphabet)
    train_data_alpha, dev_data_alpha, test_data_alpha = data_loader_alphabet.dataLoader()
    print("train sentence {}, dev sentence {}, test sentence {}.".format(
        len(train_data_alpha), len(dev_data_alpha), len(test_data_alpha)))

    # 组装 batch
    create_iter = Iterators(batch_size=[config.batch_size, config.dev_batch_size, config.test_batch_size],
                            data=[train_data_alpha, dev_data_alpha, test_data_alpha],
                            alphabet=alphabet, config=config)
    train_iter, dev_iter, test_iter = create_iter.createIterator()

    # 可选：保存 pkl 中间态，默认关。如需开启请把 config 里 save_pkl 设 True，
    # 但注意 pkl 文件含 alphabet/embed，体积可能在百 MB 级。
    if getattr(config, "save_pkl", False) and config.save_pkl is True:
        os.makedirs(config.pkl_directory, exist_ok=True)
        torch.save({"alphabet": alphabet},
                   f=os.path.join(config.pkl_directory, config.pkl_alphabet))
        torch.save({"train_iter": train_iter, "dev_iter": dev_iter, "test_iter": test_iter},
                   f=os.path.join(config.pkl_directory, config.pkl_iter))

    return train_iter, dev_iter, test_iter, alphabet


# ---------------------------------------------------------------------------
# 词表落盘
# ---------------------------------------------------------------------------
def save_dict2file(d, path):
    """
    把 dict 以 "key\tvalue\n" 格式写到文件。
    UTF-8 显式声明，避免 Windows 默认 GBK 编码踩中文。
    """
    print("Saving dictionary -> {}".format(path))
    with open(path, mode="w", encoding="utf-8") as f:
        for word, index in d.items():
            f.write("{}\t{}\n".format(word, index))


def save_dictionary(config):
    """
    把 word_alphabet / ext_word_alphabet 直接写到 save_dir/dictionary/ 下。
    旧版会在全局 dict_directory 写一份再 copytree 到 save_dir，重复且容易跨实验互相污染。
    """
    if not getattr(config, "save_dict", True):
        return

    dict_dir = os.path.join(config.save_dir, "dictionary")
    if os.path.exists(dict_dir):
        shutil.rmtree(dict_dir)
    os.makedirs(dict_dir, exist_ok=True)

    config.word_dict_path = os.path.join(dict_dir, config.word_dict)
    config.ext_word_dict_path = os.path.join(dict_dir, config.word_dict + "_ext.txt")
    save_dict2file(config.alphabet.word_alphabet.words2id, config.word_dict_path)
    save_dict2file(config.alphabet.ext_word_alphabet.words2id, config.ext_word_dict_path)


# ---------------------------------------------------------------------------
# 预训练词向量
# ---------------------------------------------------------------------------
def pre_embed(config, alphabet):
    """
    根据 config 选 zero / avg / uniform / nn 四种 OOV 处理策略加载预训练向量。
    若 pretrained_embed=False 则返回 None，由 ParserModel 自行随机初始化 ext_word_embed。
    """
    if not config.pretrained_embed:
        print("[INFO] pretrained_embed=False, skip loading pretrained vectors.")
        return None

    embed_types = ""
    if config.zeros:
        embed_types = "zero"
    elif config.avg:
        embed_types = "avg"
    elif config.uniform:
        embed_types = "uniform"
    elif config.nnembed:
        embed_types = "nn"
    else:
        embed_types = "avg"  # 默认 avg

    if not os.path.exists(config.pretrained_embed_file):
        raise FileNotFoundError(
            "预训练词向量文件不存在: {}\n"
            "请把文件放好后再训练，或在 config.cfg 关闭 [Embed] pretrained_embed".format(
                config.pretrained_embed_file)
        )

    p = Embed(path=config.pretrained_embed_file,
              words_dict=alphabet.ext_word_alphabet.id2words,
              embed_type=embed_types, pad=PAD)
    pretrain_embed = p.get_embed()
    return pretrain_embed


# ---------------------------------------------------------------------------
# 优化器名称
# ---------------------------------------------------------------------------
def get_learning_algorithm(config):
    """
    旧版用 adam=True/sgd=True 两个布尔位互斥，逻辑脆弱；这里加显式校验。
    若两个都 True 或都 False，按 adam > sgd 优先级取一个并打印警告。
    """
    if config.adam and config.sgd:
        print("[WARN] config 中 adam 和 sgd 同时为 True，强制使用 Adam。")
        return "Adam"
    if config.adam:
        return "Adam"
    if config.sgd:
        return "SGD"
    print("[WARN] config 中 adam/sgd 都为 False，默认用 Adam。")
    return "Adam"


# ---------------------------------------------------------------------------
# 把数据集大小、词表大小等运行时信息塞回 config
# ---------------------------------------------------------------------------
def get_params(config, alphabet):
    config.learning_algorithm = get_learning_algorithm(config)

    config.embed_num = alphabet.word_alphabet.vocab_size
    config.ext_embed_num = alphabet.ext_word_alphabet.vocab_size
    config.tag_num = alphabet.tag_alphabet.vocab_size
    config.rel_size = alphabet.rel_alphabet.vocab_size
    config.word_PADID = alphabet.word_PADID
    config.ext_word_PADID = alphabet.ext_word_PADID
    config.tag_PADID = alphabet.tag_PADID
    config.word_ROOTID = alphabet.word_ROOTID
    config.alphabet = alphabet

    print("embed_num : {}, ext_embed_num : {}, tag_num : {}, rel_size : {}".format(
        config.embed_num, config.ext_embed_num, config.tag_num, config.rel_size))
    print("word_PADID {}, ext_word_PADID {}, tag_PADID {}, word_ROOTID {}".format(
        config.word_PADID, config.ext_word_PADID, config.tag_PADID, config.word_ROOTID))
    print_common()


# ---------------------------------------------------------------------------
# 模型实例化
# ---------------------------------------------------------------------------
def load_model(config):
    """
    实例化 ParserModel 并搬到指定 device；test 模式下尝试从 ckpt 恢复。
    """
    print("***************************************")
    model = ParserModel(config)

    if config.device != cpu_device:
        model = model.to(config.device)

    if config.test is True:
        model = load_test_model(model, config)

    parser = BiaffineParser(model, config.word_ROOTID)
    print(model)
    print("Parser ready.")
    return parser


# ---------------------------------------------------------------------------
# 数据入口（外部唯一调用点）
# ---------------------------------------------------------------------------
def load_data(config):
    """
    现在只支持 process=True 路径。
    旧版的 process=False（从 pkl 恢复）依赖 save_pkl，但默认就被关闭，
    保留这条分支只会带来未测试的代码路径，所以删除以减少出错面。
    """
    print("load data (always process from raw conll).")
    start_time = time.time()
    train_iter, dev_iter, test_iter, alphabet = preprocessing(config)
    config.pretrained_weight = pre_embed(config=config, alphabet=alphabet)
    end_time = time.time()
    print("Load Data Use Time {:.4f}".format(end_time - start_time))
    print("***************************************")
    return train_iter, dev_iter, test_iter, alphabet
