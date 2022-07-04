'''
This file is the input program of crop segmentation network.
Author: Licong Liu
Date: 2022/7/4
Email: 543476459@qq.com
'''
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import os, torch, UNIT
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from dataset.dataset_anhui import DatasetAH
from model import unet


class TanimotoLoss(nn.Module):
    def __init__(self):
        super(TanimotoLoss, self).__init__()

    def tanimoto(self, x, y):
        return (x * y).sum(1) / ((x * x).sum(1) + (y * y).sum(1) - (x * y).sum(1))

    def forward(self, pre, tar):
        '''
        pre and tar must have same shape. (N, C, H, W)
        '''
        # 获取每个批次的大小 N
        N = tar.size()[0]
        # 将宽高 reshape 到同一纬度
        input_flat = pre.view(N, -1)
        targets_flat = tar.view(N, -1)

        t = 1 - self.tanimoto(input_flat, targets_flat)
        loss = t
        # t_ = self.tanimoto(1 - input_flat, 1 - targets_flat)
        # loss = t + t_
        loss = loss.mean()
        return loss


def acc_metric(predb, yb):
    predb = (predb > 0.5).float()
    acc = (predb == yb).float().mean()
    return acc


def f1_metric(predb, yb):
    predb = (predb > 0.5).float()

    tp = (yb * predb).sum().to(torch.float32)
    tn = ((1 - yb) * (1 - predb)).sum().to(torch.float32)
    fp = ((1 - yb) * predb).sum().to(torch.float32)
    fn = (yb * (1 - predb)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1


ROOT_DATASET = r"D:\CropSegmentation\data\AH\Training"
ROOT_PTN = r"D:\CropSegmentation\pths"
ROOT_RST = r"D:\CropSegmentation\result"


def tranining():
    ROOT_SPC = os.path.join(ROOT_DATASET, "Spectral")
    ROOT_LABEL = os.path.join(ROOT_DATASET, "Label")

    datasetah = DatasetAH(ROOT_SPC, ROOT_LABEL)

    model = unet.UNet(4).cuda()
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = TanimotoLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    print("Data Loading...")
    trainloader = torch.utils.data.DataLoader(datasetah, batch_size=4, shuffle=True, num_workers=0)
    print("Loading Finished.")
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(200):
            for step, data in enumerate(trainloader, 0):
                image, mask = data
                image = image.to(torch.float32).cuda()
                mask = mask.to(torch.float32).cuda()

                optimizer.zero_grad()
                outputs = model(image)

                loss = loss_fn(outputs, mask)
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    print('Current epoch-step: {}-{}  Loss: {}  AllocMem (Mb): {}'
                          .format(epoch, step, loss,
                                  torch.cuda.memory_allocated() / 1024 / 1024))

            scheduler.step()
            if epoch % 10 == 0:
                torch.save(model.state_dict(), "model_state_e{}.pth".format(epoch))
                print("---Validation e {}: ".format(epoch), end=", ")
                num = 0
                f_score = 0
                acc = 0
                with torch.no_grad():
                    for step, data in enumerate(trainloader, 0):
                        image, mask = data
                        image = image.to(torch.float32).cuda()
                        mask = mask.to(torch.float32).cuda()
                        outputs = model(image)
                        f_score += f1_metric(outputs, mask)
                        acc += acc_metric(outputs, mask)
                        num += 1
                print("F1 score: ", f_score/num, "; Accuracy: ", acc/num, "LR: ", scheduler.get_last_lr())


def inference(img, Model, name, interval=16, start=16, mask=None):
    '''
    将整个影像分割为各个255 * 255的小块，推测完后加回去。
    '''

    model = Model.cuda()
    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint)

    size = 256
    b, xlen, ylen = img.shape
    half_size = int(size / 2)
    x_center = np.arange(start, xlen, interval, dtype=int)
    y_center = np.arange(start, ylen, interval, dtype=int)
    x_center, y_center = np.meshgrid(x_center, y_center)

    img_count = np.zeros((xlen, ylen), dtype=np.int16)
    img_label = np.zeros((xlen, ylen), dtype=np.float32)

    xlen_chip, ylen_chip = x_center.shape
    with torch.autograd.set_detect_anomaly(True) and torch.no_grad():
        for i in tqdm(range(xlen_chip)):
            for j in range(ylen_chip):
                xloc0, xloc1 = max((x_center[i, j] - half_size, 0)), min((x_center[i, j] + half_size, xlen))
                yloc0, yloc1 = max((y_center[i, j] - half_size, 0)), min((y_center[i, j] + half_size, ylen))
                subset_img = np.zeros((b, size, size), dtype=np.float32)
                xsize, ysize = xloc1 - xloc0, yloc1 - yloc0
                subset_img[:, :xsize, :ysize] = img[:, xloc0:xloc1, yloc0:yloc1]
                if np.max(subset_img) == 0:
                    continue
                no_value_loc = np.where(np.mean(subset_img, axis=0) == 0, 0, 1)

                subset_img = np.expand_dims(subset_img, 0)
                subset_img_torch = torch.from_numpy(subset_img).cuda()

                img_sub_label = model(subset_img_torch)  # critical
                img_sub_label = img_sub_label.cpu().numpy().astype(float)
                img_sub_label *= no_value_loc

                if mask is not None:
                    img_count[xloc0:xloc1, yloc0:yloc1] += mask[:xsize, :ysize]
                    img_label *= mask
                else:
                    img_count[xloc0:xloc1, yloc0:yloc1] += 1
                img_label[xloc0:xloc1, yloc0:yloc1] += img_sub_label[0, 0, :xsize, :ysize]
    epsilon = 1e-7
    img_label = img_label / (img_count + epsilon)
    return img_label


if __name__ == "__main__":
    tranining()

    # img_subset, proj, geot = \
    #     UNIT.img2numpy(r"F:\project\Match\MatchMaterials\data\Preprocess\Subset0.tif", geoinfo=True)
    # img_label = inference(img_subset,
    #                       unet.UNet(4),
    #                       os.path.join(ROOT_PTN, "model_state_e190.pth"),
    #                       interval=32)
    # UNIT.numpy2img(os.path.join(ROOT_RST, r"img_label_predict_0704.tif")
    #                , img_label, proj=proj, geot=geot)
