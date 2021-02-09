import torch
import torch.nn as nn
import os

from torch.utils.data import DataLoader, random_split
class CUTIE(nn.Module):

    def __init__(self):
        self.OUT_CHANNELS = 32
        self.EMBEDDING_SIZE = EMBEDDING_SIZE
        self.NB_CLASS = NB_CLASS

        # grid
        self.embedding_layer = nn.Embedding(VOCAB_SIZE, EMBEDDING_SIZE)
        self.avgPool = nn.AdaptiveAvgPool2d((64, 64))
        self.avgPoolFinal = nn.AdaptiveAvgPool2d((IMAGE_SIZE, IMAGE_SIZE))

        self.conv1_1 = nn.Conv2d(self.EMBEDDING_SIZE + self.OUT_CHANNELS, self.EMBEDDING_SIZE + self.OUT_CHANNELS, 1, 1)
        self.conv1_1_layer_2 = nn.Conv2d(4 * self.OUT_CHANNELS, 4 * self.OUT_CHANNELS, 1, 1)
        self.conv_layer_3 = nn.Conv2d(5 * self.OUT_CHANNELS, 16, 3, 1, 1)
        self.conv1_1_layer_4 = nn.Conv2d(16, self.NB_CLASS, 1, 1)

        # image
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, self.OUT_CHANNELS, 3, stride=1),
            nn.BatchNorm2d(self.OUT_CHANNELS),
            nn.ReLU()
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.OUT_CHANNELS, self.OUT_CHANNELS, 3, stride=1),
            nn.BatchNorm2d(self.OUT_CHANNELS),

            nn.ReLU(),

            nn.Conv2d(self.OUT_CHANNELS, self.OUT_CHANNELS, 3, stride=1),
            nn.BatchNorm2d(self.OUT_CHANNELS),
            nn.ReLU(),

            nn.Conv2d(self.OUT_CHANNELS, self.OUT_CHANNELS, 3, stride=2),
            nn.BatchNorm2d(self.OUT_CHANNELS),
            nn.ReLU()
        )

        self.atrous_block = nn.Sequential(
            nn.Conv2d(self.OUT_CHANNELS + self.EMBEDDING_SIZE, self.OUT_CHANNELS, 3, stride=1, padding=0, dilation=2),
            nn.ReLU(),

            nn.Conv2d(self.OUT_CHANNELS, self.OUT_CHANNELS, 3, stride=1, padding=0, dilation=2),
            nn.ReLU(),

            nn.Conv2d(self.OUT_CHANNELS, self.OUT_CHANNELS, 3, stride=1, padding=0, dilation=2),
            nn.ReLU(),

            nn.Conv2d(self.OUT_CHANNELS, self.OUT_CHANNELS, 3, stride=1, padding=0, dilation=2),
            nn.BatchNorm2d(self.OUT_CHANNELS),
            nn.ReLU()

        )

        # aspp
        self.aspp_layer_1 = nn.Conv2d(self.OUT_CHANNELS, self.OUT_CHANNELS, 7, stride=1, dilation=4)
        self.aspp_layer_2 = nn.Conv2d(self.OUT_CHANNELS, self.OUT_CHANNELS, 5, padding=1, dilation=8)
        self.aspp_layer_3 = nn.Conv2d(self.OUT_CHANNELS, self.OUT_CHANNELS, 3, padding=1, dilation=16)


    def forward(self, images, grids):
        # grid
        embed = self.embedding_layer(grids)
        embed_transpose = torch.transpose(embed, 2, 3)
        embed_transpose = torch.transpose(embed_transpose, 1, 2)

        # image
        out_conv_layer_1 = self.conv_layer_1(images)

        out_conv_block = self.conv_block(out_conv_layer_1)

        embed_and_conv = torch.cat((self.avgPool(embed_transpose), self.avgPool(out_conv_block)), 1)

        embed_and_conv = self.conv1_1(embed_and_conv)

        out_atrous_block = self.atrous_block(embed_and_conv)

        out_aspp_0 = self.avgPool(out_atrous_block)
        out_aspp_1 = self.avgPool(self.aspp_layer_1(out_atrous_block))
        out_aspp_2 = self.avgPool(self.aspp_layer_2(out_atrous_block))
        out_aspp_3 = self.avgPool(self.aspp_layer_3(out_atrous_block))

        aspp = torch.cat([out_aspp_0, out_aspp_1, out_aspp_2, out_aspp_3], dim=1)

        aspp = self.avgPoolFinal(aspp)
        aspp = self.conv1_1_layer_2(aspp)
        first_layer = self.avgPoolFinal(out_conv_layer_1)

        aspp_and_first_layer = torch.cat([first_layer, aspp], dim=1)

        aspp_and_first_layer = self.conv_layer_3(aspp_and_first_layer)
        y_hat = self.conv1_1_layer_4(aspp_and_first_layer)

        return y_hat


def eval_net(net, evalloader):
    """
    Function to evaluate the model on unseen data
    net: Network CUTIE
    evalloader: Dataloader containig data for validation
    """

    net.eval()
    loss_eval = 0

    for i, data in enumerate(evalloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, grids, masks = data

        with torch.no_grad():
            # forward
            outputs = net(images.to(device), grids.to(device))
            loss = criterion(outputs.to(device), masks.squeeze(1))

            loss_eval += loss.item()
        net.train()
        return loss_eval / len(evalloader)


def apply_transformations(im, grid, mask):
    img_trans = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    mask_trans = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # transforms.ToTensor()

    ])

    grid_trans = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # transforms.ToTensor()

    ])

    return img_trans(im), grid_trans(grid), mask_trans(mask)


def get_boxes(xml_path):
    tree = ElementTree.parse(xml_path)
    parsed = tree.getroot()

    boxes = []

    for obj in parsed.findall("object"):
        xmin = int(obj.find("bndbox/xmin").text)
        ymin = int(obj.find("bndbox/ymin").text)
        xmax = int(obj.find("bndbox/xmax").text)
        ymax = int(obj.find("bndbox/ymax").text)

        boxes.append([xmin, ymin, xmax, ymax])
    return boxes


def get_mask(boxes, w, h):
    mask = np.zeros((h, w))
    positive_label = 1

    for xmin, ymin, xmax, ymax in boxes:
        mask[ymin: ymax, xmin: xmax] = positive_label

    return mask