# -*-coding:utf-8 -*-
'''
author:侯清刚
datetime:2023年06月11日
'''

import time
from torch.optim.lr_scheduler import StepLR
import os
import sys
import json
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
import logging
from tensorboardX import SummaryWriter
from ModifiedStudentModel_B_M_C_W_5_new import *

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
    logger = get_logger('./ModifiedStudentModel_B_M_C_W_5_new.log')
    logger.info('start training!')
    tb_writer = SummaryWriter('ModifiedStudentModel_B_M_C_W_5_new')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    img_size=224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path

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
    with open('./class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 64
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
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






    # 创建学生模型实例
    student_model = ModifiedStudentModel_skip()
    # student_model = ModifiedStudentModel_dilated()  #多尺度可分离卷积参数量相对更更少
    # student_model = ModifiedStudentModel_dilated_double()


     #加载权重
    weights="../My_model_Dataset_skip_win2_0.7.pth"
    # weights=''
    if weights != "":
        assert os.path.exists(weights), "weights file: '{}' not exist.".format(weights)
        weights_dict = torch.load(weights, map_location=device)
        weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "classifier" in k:
                del weights_dict[k]
        print(student_model.load_state_dict(weights_dict, strict=False))

    # 定义损失函数和优化器
    criterion1 = nn.CrossEntropyLoss()

    # 是否冻结权重
    freeze_layers=True
    if freeze_layers:
        for name, para in student_model.named_parameters():
            # 除最后一个卷积层和全连接层外，其他权重全部冻结
            if ("features.top" not in name) and ("classifier" not in name):
            # if  ("classifier" not in name):
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=0.001, momentum=0.9, weight_decay=1E-4)
    # optimizer = optim.Adam(pg, lr=0.001)

    # 定义学习率衰减策略
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)


    student_model.to(device)




    start_epoch=0
    epochs = 100
    best_acc = 0.0
    save_path = './ModifiedStudentModel_B_M_C_W_5_new.pth'
    train_steps = len(train_loader)
    checkpoint_interval=20
    RESUME=0
    if RESUME:

        path_checkpoint = "./checkpoint_59_epoch_ModifiedStudentModel_B_M_C_W_5_new.pth"
        checkpoint = torch.load(path_checkpoint)

        student_model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch']
        print("start_epoch:{}".format(start_epoch))

        total_time=0
        for epoch in range(start_epoch,epochs):
            # train
            scheduler.step()
            student_model.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                student_outputs = student_model(images.to(device))

                # 计算分类损失
                classification_loss = criterion1(student_outputs, labels.to(device))



                loss = classification_loss

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)


            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint = {"model_state_dict": student_model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "epoch": epoch}
                path_checkpoint = "./checkpoint_{}_epoch_ModifiedStudentModel_B_M_C_W_5_new.pth".format(epoch)
                torch.save(checkpoint, path_checkpoint)


            # validate
            student_model.eval()
            start_time=time.time()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = student_model(val_images.to(device))  # eval model only have last output layer
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            end_time=time.time()
            total_time+=end_time-start_time

            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(student_model.state_dict(), save_path)

            tags = ["loss", "accuracy","learning_rate"]
            tb_writer.add_scalar(tags[0], running_loss/train_steps, epoch)
            tb_writer.add_scalar(tags[1], val_accurate, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            logger.info('Epoch:[{}/{}]\t train_loss={:.5f}\t val_accurate={:.3f}'.format(epoch, epochs, running_loss / train_steps, val_accurate))

        average_time = total_time / (epochs - start_epoch)
        logger.info('average_time={:.3f}'.format(average_time))
        logger.info('finish training!')
    else:
        total_time = 0
        for epoch in range(start_epoch, epochs):
            # train
            scheduler.step()
            student_model.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data
                optimizer.zero_grad()
                student_outputs = student_model(images.to(device))

                # 计算分类损失
                classification_loss = criterion1(student_outputs, labels.to(device))

                loss = classification_loss

                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

                train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)

            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint = {"model_state_dict": student_model.state_dict(),
                              "optimizer_state_dict": optimizer.state_dict(),
                              "epoch": epoch}
                path_checkpoint = "./checkpoint_{}_epoch_ModifiedStudentModel_B_M_C_W_5_new.pth".format(epoch)
                torch.save(checkpoint, path_checkpoint)

            # validate
            student_model.eval()
            start_time = time.time()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(validate_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = student_model(val_images.to(device))  # eval model only have last output layer
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            end_time = time.time()
            total_time += end_time - start_time

            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(student_model.state_dict(), save_path)

            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], running_loss / train_steps, epoch)
            tb_writer.add_scalar(tags[1], val_accurate, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            logger.info('Epoch:[{}/{}]\t train_loss={:.5f}\t val_accurate={:.3f}'.format(epoch, epochs,
                                                                                         running_loss / train_steps,
                                                                                         val_accurate))

        average_time = total_time / (epochs - start_epoch)
        logger.info('average_time={:.3f}'.format(average_time))
        logger.info('finish training!')

    print('Finished Training')


if __name__ == '__main__':
    main()



