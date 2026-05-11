# @Author : bamtercelboo
# @Datetime : 2019/01/14 15:00
# @File : DataLoader.py
# @Last Modify Time : 2018/1/30 15:58
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    文件说明：依存句法数据读取与批处理工具
    功能：提供文本清洗、样本排序、依存树批量构造以及数据集加载
"""
import os
import sys
import re
import random
import json
import time
import torch
import numpy as np
# from sklearn.externals import joblib
from collections import OrderedDict
from Dataloader.Instance import Instance

from Dataloader.Dependency import readDepTree
from Dataloader.Dependency import *

from DataUtils.Common import *
# 统一随机种子，保证实验可复现
torch.manual_seed(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)


def batch_variable_depTree(trees, heads, rels, lengths, alphabet):
    """
    将批量的依存树信息拼装为 Dependency 序列。
    :param trees: 依存树节点列表的批量
    :param heads: 依存头索引的批量
    :param rels: 依存关系标签 id 的批量
    :param lengths: 句长批量
    :param alphabet: 标签映射字典
    :return: 逐句输出 Dependency 序列的生成器
    """
    for tree, head, rel, length in zip(trees, heads, rels, lengths):
        sentence = []
        for idx in range(length):
            # 逐词构造依存节点，关联 head 与关系标签
            sentence.append(Dependency(idx, tree[idx].org_form, tree[idx].tag, head[idx], alphabet.rel_alphabet.id2words[rel[idx]]))
        yield sentence


class DataLoaderHelp(object):
    """
    数据加载辅助函数集合
    """

    @staticmethod
    def _clean_str(string):
        """
        英文文本清洗：保留常见符号并统一空白。
        参考实现改编自 yoonkim/CNN_sentence 的预处理脚本。
        """
        # 非法字符替换为空格
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        # 标点前后补空格，便于分词
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        # 合并多余空白
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def _clean_punctuation(string):
        """
        中文标点清洗：去除常见标点并统一大小写。
        :param string: 原始文本
        :return: 清洗后的文本
        """
        # 保留英文所有格与缩写
        string = re.sub(r"\'s", " \'s", string)
        # 删除中文标点
        string = re.sub(r"，", "", string)
        string = re.sub(r"。", "", string)
        string = re.sub(r"“", "", string)
        string = re.sub(r"”", "", string)
        string = re.sub(r"、", "", string)
        string = re.sub(r"：", "", string)
        string = re.sub(r"；", "", string)
        string = re.sub(r"（", "", string)
        string = re.sub(r"）", "", string)
        string = re.sub(r"《 ", "", string)
        string = re.sub(r"》", "", string)
        # string = re.sub(r"× ×", "", string)
        # string = re.sub(r"x")
        # 合并重复空格
        string = re.sub(r"  ", " ", string)
        return string.lower()

    @staticmethod
    def _sort(insts):
        """
        按句长从大到小排序，便于后续按长度分桶或裁剪。
        :param insts: Instance 列表
        :return: 排序后的 Instance 列表
        """
        sorted_insts = []
        sorted_dict = {}
        for id_inst, inst in enumerate(insts):
            sorted_dict[id_inst] = inst.words_size
        dict = sorted(sorted_dict.items(), key=lambda d: d[1], reverse=True)
        # 根据排序后的索引重新组织样本
        for key, value in dict:
            sorted_insts.append(insts[key])
        print("Sort Finished.")
        return sorted_insts


class DataLoader(DataLoaderHelp):
    """
    负责读取依存句法数据并组装为 Instance 列表
    """
    def __init__(self, path, shuffle, config, alphabet=None):
        """
        :param path: 数据路径列表
        :param shuffle: 是否打乱
        :param config: 配置对象
        :param alphabet: 字典与标签映射
        """
        #
        print("Loading Data......")
        self.data_list = []
        self.max_count = config.max_count
        self.path = path
        self.shuffle = shuffle
        self.alphabet = alphabet

    def dataLoader(self):
        """
        批量读取指定路径列表中的数据集。
        :return: 按路径顺序返回训练/验证/测试集合
        """
        start_time = time.time()
        path = self.path
        shuffle = self.shuffle
        assert isinstance(path, list), "Path Must Be In List"
        print("Data Path {}".format(path))
        for id_data in range(len(path)):
            print("Loading Data Form {}".format(path[id_data]))
            # 逐路径读取数据
            insts = self._Load_Each_JsonData(path=path[id_data])
            print("shuffle data......")
            # 打乱样本顺序
            random.shuffle(insts)
            self.data_list.append(insts)
        end_time = time.time()
        print("DataLoader Time {:.4f}".format(end_time - start_time))
        # return train/dev/test data
        if len(self.data_list) == 3:
            return self.data_list[0], self.data_list[1], self.data_list[2]
        elif len(self.data_list) == 2:
            return self.data_list[0], self.data_list[1]

    def _Load_Each_JsonData(self, path=None, train=False):
        """
        读取单个数据文件并构造 Instance 列表。
        :param path: 数据文件路径
        :param train: 预留参数（未使用）
        :return: Instance 列表
        """
        assert path is not None, "The Data Path Is Not Allow Empty."
        insts = []
        now_lines = 0
        print()
        with open(path, encoding="UTF-8") as inf:
            for sentence in readDepTree(inf, self.alphabet):
                now_lines += 1
                if now_lines % 2000 == 0:
                    # 读取进度提示
                    sys.stdout.write("\rreading the {} line\t".format(now_lines))
                # 将句子封装为 Instance
                inst = Instance()
                inst.sentence = sentence
                insts.append(inst)
                if len(insts) == self.max_count:
                    # 达到上限后提前停止
                    break
            sys.stdout.write("\rreading the {} line\t".format(now_lines))
        return insts

