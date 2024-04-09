import os
import argparse

# from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import torch
import torchvision.datasets as datasets
from torch import nn
from torch.utils import data
from torchvision.transforms import transforms

from model.utils import train_one_epoch, evaluate

# from model.transform import Transforms
from gsdnet import GSDNet

parser = argparse.ArgumentParser()
# 数据集类别数量
parser.add_argument('--num_classes', type=int, default=7)
# 训练轮数
parser.add_argument('--epochs', type=int, default=40)
# batchSize大小
parser.add_argument('--batch-size', type=int, default=64)
# 学习率
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lrf', type=float, default=0.0001)
# 是否启用SyncBatchNorm
parser.add_argument('--syncBN', type=bool, default=True)

# 数据集所在根目录
parser.add_argument('--data-path', type=str, default="/data/zhujinlin/RAF-DB/data")

# 权重文件所在目录
parser.add_argument('--weights', type=str,
                    # default='./weights/affect_model.pth',
                    default='./weights/RAF_model.pth',
                    # default='',
                    help='initial weights path')
parser.add_argument('--freeze-layers', type=bool, default=False)
# 指定GPU
parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
# 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
parser.add_argument('--world-size', default=4, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
args = parser.parse_args()


def main(a,b):
    # 指定GPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # 训练集和测试集路径
    traindir = os.path.join(args.data_path, 'train')
    valdir = os.path.join(args.data_path, 'test')

    # 训练集和测试集dataset
    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose([
                                             transforms.Resize((224, 224)),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomApply([
                                                 transforms.RandomRotation(20),
                                                 transforms.RandomCrop(224, padding=32)
                                             ], p=0.2),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225]),
                                             transforms.RandomErasing(scale=(0.02, 0.25))
                                         ]))

    val_dataset = datasets.ImageFolder(valdir,
                                       transforms.Compose([transforms.Resize((224, 224)),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])
                                                           ]))


    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # 训练集和测试集loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               drop_last=True,
                                               shuffle=True,
                                               num_workers=nw,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             # drop_last=True,
                                             shuffle=False,
                                             num_workers=nw,
                                             pin_memory=True)

    # 加载模型
    model = GSDNet(num_class=args.num_classes).to(device)

    # 初始化权重
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights,map_location='cuda:0')
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # 损失函数
    criterion = {
        'softmax': nn.CrossEntropyLoss().to(device),
    }

    # 优化算法
    optimizer = {
        'softmax': torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4),
    }

    # 优化策略
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer['softmax'], gamma=0.9)
    best_acc = 0.0

    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,
                                                criterion=criterion,
                                                a=a,
                                                b=b)

        scheduler.step()
        # 测试
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch,
                                     criterion=criterion)
        # 保存模型权重文件
        if best_acc <= val_acc:
            torch.save(model.state_dict(), "./weights/RAF_model.pth")
            best_acc = val_acc
        print(val_acc)
        print(best_acc)
    return best_acc


if __name__ == '__main__':
    main(0.01, 0.3)

