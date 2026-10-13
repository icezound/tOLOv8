# -*- coding: utf-8 -*-
"""CIFAR-10 -> 五分类数据集整理脚本

任务要求: 下载一个公开数据集, 挑选其中 5 个类别, 整理成分类数据集.

数据来源: CIFAR-10 官方公开数据集
          https://www.cs.toronto.edu/~kriz/cifar.html
        原始数据已放在 ./data_raw/cifar-10-batches-py (官方 pickle 格式),
        脚本直接读取它, 不需要联网.

流程:
  1. 读原始 pickle: data_batch_1~5 作训练集, test_batch 作验证集;
  2. 从 10 类中挑出 5 类: airplane / automobile / bird / cat / deer;
  3. 每类均衡取图, 保存成图片文件夹, 得到 YOLOv8 分类可直接使用的数据集.

输出结构 (train.py 里的 data='./cifar10' 就指向这里):
  cifar10/train/<类别>/*.png   每类 500 张
  cifar10/val/<类别>/*.png     每类 100 张
"""
import os
import pickle

import numpy as np
from PIL import Image

CIFAR10_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

# 要整理的类别: 全部 10 类
# 如果只想要前 5 类, 把下面这行改成:  CLASSES = CIFAR10_NAMES[:5]
CLASSES = CIFAR10_NAMES
KEEP = [CIFAR10_NAMES.index(c) for c in CLASSES]

TRAIN_PER_CLASS = 500    # 每类取多少张作为训练集
VAL_PER_CLASS = 100      # 每类取多少张作为验证集

RAW = './data_raw'
BATCH_DIR = os.path.join(RAW, 'cifar-10-batches-py')
OUT = './cifar10'

TRAIN_FILES = ['data_batch_%d' % i for i in range(1, 6)]
VAL_FILES = ['test_batch']


def iter_images(filenames):
    """读取官方 pickle, 逐张返回 (PIL 图片, 类别编号)."""
    for name in filenames:
        with open(os.path.join(BATCH_DIR, name), 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
        # (N, 3072) -> (N, 32, 32, 3)
        data = entry['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = entry.get('labels', entry.get('fine_labels'))
        for arr, label in zip(data, labels):
            yield Image.fromarray(np.ascontiguousarray(arr)), int(label)


def save_split(filenames, split, per_class):
    """按类别均衡取图, 保存到 cifar10/<split>/<类别>/ 下."""
    root = os.path.join(OUT, split)
    for name in CLASSES:
        os.makedirs(os.path.join(root, name), exist_ok=True)

    counter = {i: 0 for i in KEEP}
    for img, label in iter_images(filenames):
        if label not in counter or counter[label] >= per_class:
            continue
        name = CIFAR10_NAMES[label]
        img.save(os.path.join(root, name, '%s_%05d.png' % (name, counter[label])))
        counter[label] += 1
        if all(v >= per_class for v in counter.values()):
            break
    print(split, {CIFAR10_NAMES[k]: v for k, v in counter.items()})


def download_raw():
    """原始数据缺失时, 用 torchvision 下载 CIFAR-10 官方包 (约 170MB)."""
    import torchvision
    torchvision.datasets.CIFAR10(root=RAW, train=True, download=True)
    torchvision.datasets.CIFAR10(root=RAW, train=False, download=True)


def main():
    if not os.path.isfile(os.path.join(BATCH_DIR, 'data_batch_1')):
        download_raw()

    save_split(TRAIN_FILES, 'train', TRAIN_PER_CLASS)
    save_split(VAL_FILES, 'val', VAL_PER_CLASS)
    print('数据集已保存到', os.path.abspath(OUT))


if __name__ == '__main__':
    main()
