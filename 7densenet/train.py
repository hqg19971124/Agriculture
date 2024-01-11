import os
import math
import argparse
import time
import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets
from model import densenet121, load_state_dict
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate
import logging


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
    logger = get_logger('./densenet.log')
    logger.info('start training!')

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
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
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
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
    model = densenet121(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            load_state_dict(model, args.weights)
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "classifier" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4, nesterov=True)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    start_epoch=0
    checkpoint_interval=50
    best_acc = 0.0
    save_path = './densenet.pth'
    RESUME=0
    if RESUME:

        path_checkpoint = "./checkpoint_3_epoch.pth"
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch']
        print("start_epoch:{}".format(start_epoch))


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
                path_checkpoint = "./checkpoint_{}_epoch.pth".format(epoch)
                torch.save(checkpoint, path_checkpoint)

            # validate
            acc = evaluate(model=model,
                           data_loader=validate_loader,
                           device=device)

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

        logger.info('finish training!')
    else:
        total_time=0
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
                path_checkpoint = "./checkpoint_{}_epoch.pth".format(epoch)
                torch.save(checkpoint, path_checkpoint)

            # validate
            start_time=time.time()
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

            logger.info('Epoch:[{}/{}]\t train_loss={:.5f}\t val_accurate={:.3f}'.format(epoch, args.epochs, mean_loss, acc))
            average_time = total_time / (args.epochs - start_epoch)
            logger.info('average_time={:.3f}'.format(average_time))
        logger.info('finish training!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    # parser.add_argument('--data-path', type=str,
    #                     default="/data/flower_photos")

    # densenet121 官方权重下载地址
    # https://download.pytorch.org/models/densenet121-a639ec97.pth
    parser.add_argument('--weights', type=str, default='densenet121.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
