import os
import sys
import json
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model_v3 import mobilenet_v3_small
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


def main():

    logger = get_logger('./exp.log')
    logger.info('start training!')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
    # image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    image_path = os.path.join(data_root, "Dataset1")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=8)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    batch_size = 4
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

    # create model
    net = mobilenet_v3_small()
    # net = MobileNetV2(num_classes=9)

    original_classifier = net.classifier
    # 修改全连接层的结构
    # 修改全连接层的结构
    num_features = net.classifier[-1].out_features
    new_classifier = torch.nn.Sequential(
        original_classifier,
        torch.nn.Linear(num_features, 9)
    )
    net.classifier = new_classifier
    # load pretrain weights
    # download url: https://download.pytorch.org/cdmodels/mobilenet_v2-b0353104.pth
    model_weight_path = "./mobilenet_v3.pth"
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location='cpu')

    # delete classifier weights
    # pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_weights, strict=False)

    # freeze features weights
    for param in net.features.parameters():
        param.requires_grad = False

    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    start_epoch=0

    epochs = 2
    best_acc = 0.0
    save_path = './MobileNetV3.pth'
    train_steps = len(train_loader)
    checkpoint_interval=5
    RESUME=0
    if RESUME:

        path_checkpoint = "./checkpoint_3_epoch.pth"
        checkpoint = torch.load(path_checkpoint)

        net.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch']
        print("start_epoch:{}".format(start_epoch))

        for epoch in range(start_epoch,epochs):
            # train
            net.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)

            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint = {"model_state_dict": net.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "epoch": epoch}
                path_checkpoint = "./checkpoint_{}_epoch.pth".format(epoch)
                torch.save(checkpoint, path_checkpoint)

            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               epochs)
            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], running_loss/train_steps, epoch)
            tb_writer.add_scalar(tags[1], val_accurate, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)


            logger.info('Epoch:[{}/{}]\t train_loss={:.5f}\t val_accurate={:.3f}'.format(epoch, epochs, running_loss / train_steps, val_accurate))

        logger.info('finish training!')
    else:
        for epoch in range(start_epoch,epochs):
            # train
            net.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)

            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint = {"model_state_dict": net.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "epoch": epoch}
                path_checkpoint = "./checkpoint_{}_epoch.pth".format(epoch)
                torch.save(checkpoint, path_checkpoint)

            # validate
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               epochs)
            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)


            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], running_loss/train_steps, epoch)
            tb_writer.add_scalar(tags[1], val_accurate, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            logger.info('Epoch:[{}/{}]\t train_loss={:.5f}\t val_accurate={:.3f}'.format(epoch, epochs, running_loss / train_steps, val_accurate))

        logger.info('finish training!')
    print('Finished Training')


if __name__ == '__main__':
    main()
