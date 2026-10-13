import os
import torchvision

# 10个类别名称
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def save_data(dataset, split):
    root = f'cifar10/{split}'
    os.makedirs(root, exist_ok=True)
    for idx, (img, label) in enumerate(dataset):
        if idx >= 500:  # 总共只取前500张，每类平均50张
            break
        # 后面保存图片的代码不变

# 自动下载训练集、验证集
train_set = torchvision.datasets.CIFAR10(root='./data_raw', train=True, download=True)
val_set = torchvision.datasets.CIFAR10(root='./data_raw', train=False, download=True)

save_data(train_set, 'train')
save_data(val_set, 'val')