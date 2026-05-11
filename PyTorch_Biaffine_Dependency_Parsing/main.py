# -*- coding: utf-8 -*-
"""
文件说明：训练 / 测试入口

本文件相对原版的工程化改造（按 CLAUDE.md 要求）：
    1. 删除 Python 3.12 已移除的 `imp.reload`，统一用 utf-8 默认编码
    2. 删除对不存在的 `test.py` 模块的 import，把推理分支占位为 NotImplementedError
       （避免 import error 让训练直接挂掉，仍保留 --test 命令行接口供未来扩展）
    3. 不再用 shutil.copytree 把 Model / DataUtils / Dataloader / Config 整个复制到 save_dir，
       改为：在 Output/<时间戳>/ 下只保留 ckpt + log + 生效配置 + history.json + 曲线图
    4. 把训练产物统一落到 config.save_direction 指定目录（默认 Output/）而非 Save_All/
    5. 加 logging（同时写文件 + stdout），替代散落的 print
    6. **不再为每个对比实验复制 cfg**：通过 `--override Section.key=value` 在命令行覆盖；
       训练时把生效配置落到 Output/<run>/effective_config.cfg，读者直接看产物即可复现
"""

import argparse
import datetime
import json
import os
import random
import sys

import torch

import Config.config as configurable
from DataUtils.Common import seed_num, cpu_device
from DataUtils.Logger import get_logger
from DataUtils.mainHelp import (
    load_data, get_params, save_dictionary, load_model
)
from trainer import Train

torch.manual_seed(seed_num)
random.seed(seed_num)


def start_train(train_iter, dev_iter, test_iter, parser, config, logger):
    t = Train(train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter,
              model=parser, config=config, logger=logger)
    t.train()
    logger.info("Finish Train.")


def start_test(*args, **kwargs):
    raise NotImplementedError(
        "独立推理脚本尚未实现；请用训练时落盘的 best.pt + Dataloader 自行推理。"
    )


def main(config, logger):
    """save_dir / ckpt_dir 已在 __main__ 中创建好；这里负责落生效配置 + meta，然后训练。"""
    # 写"生效配置"快照（基础 cfg + override 后），取代旧版复制原始 cfg 的做法
    eff_path = os.path.join(config.save_dir, "effective_config.cfg")
    config.dump_effective(eff_path)
    logger.info("Effective config -> {}".format(eff_path))

    meta = {
        "start_time": config.mulu,
        "device": config.device,
        "train": config.train,
        "test": config.test,
        "config_file": os.path.abspath(config.config_file),
        "argv": sys.argv,
        "overrides": getattr(config, "_overrides_raw", []),
    }
    with open(os.path.join(config.save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Save dir : {}".format(config.save_dir))
    logger.info("Ckpt dir : {}".format(config.ckpt_dir))

    train_iter, dev_iter, test_iter, alphabet = load_data(config=config)
    get_params(config=config, alphabet=alphabet)
    save_dictionary(config=config)
    parser = load_model(config)

    if config.train is True:
        start_train(train_iter, dev_iter, test_iter, parser, config, logger)
        return
    if config.test is True:
        start_test(train_iter, dev_iter, test_iter, parser, alphabet, config)


def parse_argument():
    parser = argparse.ArgumentParser(
        description="PyTorch Biaffine Dependency Parsing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  # 默认配置训练\n"
            "  python main.py -c ./Config/config.cfg --device cuda:0\n\n"
            "  # 不复制 cfg，直接命令行覆盖任意字段\n"
            "  python main.py --override Train.epochs=50 Optimizer.adam=False Optimizer.sgd=True\n\n"
            "  # 维度对比：基础 cfg 不动，只覆盖维度+词向量路径\n"
            "  python main.py --override Model.embed_dim=300 \\\n"
            "                            Embed.pretrained_embed_file=./Data/Embed/sgns.baidubaike.300d\n"
        ),
    )
    parser.add_argument("-c", "--config", dest="config_file", type=str,
                        default="./Config/config.cfg", help="config path")
    parser.add_argument("-device", "--device", dest="device", type=str, default="cpu",
                        help="device['cpu','cuda:0','cuda:1',...]")
    parser.add_argument("--train", dest="train", action="store_true", default=True,
                        help="train model")
    parser.add_argument("-p", "--process", dest="process", action="store_true", default=True,
                        help="process raw data on-the-fly (不读 pkl 缓存)")
    parser.add_argument("-t", "--test", dest="test", action="store_true", default=False,
                        help="test model (尚未实现独立推理)")
    parser.add_argument("--t_model", dest="t_model", type=str, default=None,
                        help="model path for test")
    parser.add_argument("--t_data", dest="t_data", type=str, default=None,
                        help="dataset name for test [train/dev/test]")
    parser.add_argument("--predict", dest="predict", action="store_true", default=False,
                        help="predict mode (尚未实现)")
    parser.add_argument("--override", dest="override", type=str, nargs="*", default=[],
                        help="覆盖配置字段，格式 Section.key=value，可多个空格分隔")
    parser.add_argument("--tag", dest="tag", type=str, default="",
                        help="实验标签，加到产物目录名末尾，方便区分对比实验")
    args = parser.parse_args()

    config = configurable.Configurable(config_file=args.config_file)
    if args.override:
        config.override(args.override)
    config._overrides_raw = list(args.override)

    config.config_file = args.config_file
    config.device = args.device
    config.train = args.train
    config.process = args.process
    config.test = args.test
    config.t_model = args.t_model
    config.t_data = args.t_data
    config.predict = args.predict
    config.tag = args.tag

    if config.test is True:
        config.train = False
    if config.t_data not in [None, "train", "dev", "test"]:
        parser.print_help()
        print("t_data : {}, not in [None, 'train', 'dev', 'test']".format(config.t_data))
        sys.exit(1)

    return config


if __name__ == "__main__":
    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    config = parse_argument()

    if config.device != cpu_device and config.device.startswith("cuda"):
        try:
            device_number = int(config.device.split(":")[-1])
            torch.cuda.set_device(device_number)
            torch.cuda.manual_seed(seed_num)
            torch.cuda.manual_seed_all(seed_num)
            print("Using GPU {} (current cuda device {})".format(
                config.device, torch.cuda.current_device()))
        except (ValueError, RuntimeError) as e:
            print("[WARN] cuda init failed ({}), fallback to CPU.".format(e))
            config.device = cpu_device

    # 产物目录命名：时间戳 + 可选 tag，方便从一堆 Output/2026-...-... 里识别哪个是哪个对比实验
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config.mulu = "{}_{}".format(ts, config.tag) if config.tag else ts
    config.save_dir = os.path.join(config.save_direction, config.mulu)
    config.ckpt_dir = os.path.join(config.save_dir, "ckpts")
    os.makedirs(config.ckpt_dir, exist_ok=True)
    log_path = os.path.join(config.save_dir, "logs", "train.log")
    logger = get_logger(log_path)

    logger.info("CMD: {}".format(" ".join(sys.argv)))
    logger.info("Device : {}".format(config.device))
    logger.info("Train model : {}".format(config.train))
    logger.info("Tag : {}".format(config.tag or "(none)"))

    main(config, logger)
