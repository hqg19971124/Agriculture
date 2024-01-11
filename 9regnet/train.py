import os
import math
import argparse
from torch.optim.lr_scheduler import StepLR
import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets
from model_regnet import create_regnet
from utils import read_split_data, train_one_epoch, evaluate
import logging
import time

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def main(args):
    logger = get_logger('./regnet.log')
    logger.info('start training!')

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter('regnet_dataset1')
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    image_path = os.path.join(data_root, "Dataset")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])

    train_num = len(train_dataset)




    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # 如果存在预训练权重则载入
    model = create_regnet(model_name=args.model_name,
                          num_classes=args.num_classes).to(device)
    # print(model)

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("train {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    start_epoch = 0
    best_acc = 0.0
    checkpoint_interval = 20
    save_path = './regnet_Dataset.pth'
    resume = 0
    if resume:
        path_checkpoint = "./checkpoint_3_epoch.pth"
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch']
        print("start_epoch:{}".format(start_epoch))

        total_time=0
        for epoch in range(start_epoch,args.epochs):
            # train
            mean_loss = train_one_epoch(model=model,
                                        optimizer=optimizer,
                                        data_loader=train_loader,
                                        device=device,
                                        epoch=epoch)

            scheduler.step()

            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint = {"model_state_dict": model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "epoch": epoch}
                path_checkpoint = "./checkpoint_{}_epoch_Dataset1.pth".format(epoch)
                torch.save(checkpoint, path_checkpoint)

            # validate
            start_time=time.time()
            acc = evaluate(model=model,
                           data_loader=validate_loader,
                           device=device)
            end_time=time.time()
            total_time+=end_time-start_time
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], acc, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)
            logger.info('Epoch:[{}/{}]\t train_loss={:.5f}\t val_accurate={:.3f}'.format(epoch, args.epochs, mean_loss, acc))

        average_time = total_time / (args.epochs - start_epoch)
        logger.info('average_time={:.3f}'.format(average_time))
        logger.info('finish training!')
    else:
        total_time = 0
        for epoch in range(start_epoch, args.epochs):
            # train
            mean_loss = train_one_epoch(model=model,
                                        optimizer=optimizer,
                                        data_loader=train_loader,
                                        device=device,
                                        epoch=epoch)

            scheduler.step()

            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint = {"model_state_dict": model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "epoch": epoch}
                path_checkpoint = "./checkpoint_{}_epoch_Dataset1.pth".format(epoch)
                torch.save(checkpoint, path_checkpoint)

            # validate
            start_time = time.time()
            acc = evaluate(model=model,
                           data_loader=validate_loader,
                           device=device)
            end_time = time.time()
            total_time += end_time - start_time
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], acc, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)
            logger.info(
                'Epoch:[{}/{}]\t train_loss={:.5f}\t val_accurate={:.3f}'.format(epoch, args.epochs, mean_loss, acc))

        average_time = total_time / (args.epochs - start_epoch)
        logger.info('average_time={:.3f}'.format(average_time))
        logger.info('finish training!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="/data/flower_photos")
    parser.add_argument('--model-name', default='RegNetY_400MF', help='create model name')

    # 预训练权重下载地址
    # 链接: https://pan.baidu.com/s/1XTo3walj9ai7ZhWz7jh-YA  密码: 8lmu
    parser.add_argument('--weights', type=str, default='regnety_400mf.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
