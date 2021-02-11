import torch
from torch.utils.data import Dataset
from gensim import corpora
from matplotlib import cm
from PIL import Image
import numpy as np
import os
import json
import cv2
import pytesseract


def build_bboxes(path_receipt_json):
    """ Get the cordinates of the bounding boxes from the jsonf file.
    parameters :
    -----------
        - path_receipt_json : path to the json file of the receipt.
    """
    with open(path_receipt_json,"r") as json_file :
      receipt_json= json.load(json_file)
    # Get the cordinates of the bounding boxes from the jsonf file
    BoundingBoxes  = []
    for block in recept_json["text_boxes"]:
        x,y,w,h = block["bbox"]
        # In the paper, the coordinates of the top left and bottom right are used instead of the coordinates of the top left and the width and the height.
        BoundingBoxes.append([x,y, x+w,y+h])
    return BoundingBoxes


def build_mask(BoundingBoxes, image_width, image_height):
    """ highlight the pixels that are the bounding boxes in the images. Theses pixels are given 1 and the rest is zero.
        This process is done for each bbox in the image.
        parameters :
        -----------
            - BoundingBoxes : the list of the boounding boxes coordinates.
            - image_width   : the width of the receipt image.
            - image_height  : the height of the receipt image
     """
    # First everything is set to zero.
    mask = np.zeros((image_height, image_width))
    # Then the triangles of the bounding boxes are changed to 1.
    for x_top_left,y_top_left,x_bottom_right,y_bottom_right in BoundingBoxes:
        mask[y_top_left: y_bottom_right, x_top_left: x_bottom_right] = 1
    return mask




def build_Grid(image, gensim_dictionary):
    """ From the json file build the grid to feed the model """
    # The number of pixels between two tokens (Horizontally)
    STRIDE = 4

    image = np.array(image)
    img_h, img_w, _ = image.shape

    grid = np.zeros((img_h, img_w))

    custom_config = r'-l eng+it --psm 6'

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    text = pytesseract.image_to_string(thresh, config=custom_config)

    line_text_list = text.split("\n")
    line_text_list_tokenized = [word_tokenize(line) for line in line_text_list]

    gensim_dictionary.add_documents(line_text_list_tokenized)

    nb_sentences = len(line_text_list_tokenized)
    nb_token_per_line = [len(line) for line in line_text_list_tokenized]

    bboxes_h = int((img_h - 2 * nb_sentences) / nb_sentences)
    bboxes_per_line_w = ((img_w - 2 * np.array(nb_token_per_line)) / np.array(nb_token_per_line)).astype(int)
    # Building the grid
    line_cursor = 0
    for line_i in range(nb_sentences):
        col_idx = 0

        for token_i in line_text_list_tokenized[line_i]:
            grid[line_cursor:line_cursor + bboxes_h, col_idx:col_idx + bboxes_per_line_w[line_i]] = token2id(
                gensim_dictionary, token_i)
            col_idx += (bboxes_per_line_w[line_i] + STRIDE)

        line_cursor += (bboxes_h + STRIDE)

    return grid, gensim_dictionary




class Images_Labels_Dataset(Dataset):
    """ Receipts and the correspondings grids dataset. """

    def __init__(self, receipts_image_dir, receipt_JSON_dir, transforms=None):
        """
        Args:
            receipts_image_dir : Directory with all the images.
            receipt_JSON_dir   : directory of the receipts in json format after the OCR  been made.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.receipts_image_dir = receipts_image_dir
        self.receipt_JSON_dir = receipt_JSON_dir
        self.transforms = transforms
        # preprare the  path for each receipt  in the root directory.
        receipts_images = os.listdir(receipts_image_dir)
        receipts_images.sort()
        self.receipts_images = [os.path.join(receipts_image_dir, one_receipt_path) for one_receipt_path in receipts_images]

        # load bounding coordinates
        receipts_JSON = os.listdir(receipt_JSON_dir)
        receipts_JSON.sort()
        self.receipts_json = [os.path.join(receipt_JSON_dir, one_json_path) for one_json_path in receipts_JSON]

        len_min = min(len(self.receipts_images), len(self.receipts_json))
        self.receipts_images = self.receipts_images[:len_min]
        self.receipts_json = self.receipts_json[:len_min]

        self.dictionary = corpora.Dictionary([])

    def __getitem__(self, idx):

        """ Read one or a list receipt image and the correspondent json file

        parameters :
        -----------
            - idx : index of the image to read in the directory containing the images and the json files.
          """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        one_receipt_image = Image.open(self.receipts_images[idx]).convert("RGB")
        w, h = one_receipt_image.size

        # 0: other, 1: total: boxes
        boxes = build_bboxes(self.receipts_json[idx])
        grid, self.dictionary = build_Grid(one_receipt_image, self.dictionary)
        mask = build_mask(boxes, w, h)

        mask = Image.fromarray(np.uint8(cm.gist_earth(mask) * 255)).convert("L")

        grid = Image.fromarray(np.uint8(cm.gist_earth(grid) * 255)).convert("L")

        if self.transforms is not None:
            img, grid, mask = self.transforms(one_receipt_image, grid, mask)

        mask = np.where(np.array(mask) > 0, 1, 0)

        image = torch.as_tensor(np.array(one_receipt_image), dtype=torch.float32)
        mask = torch.as_tensor(mask, dtype=torch.long).unsqueeze(0)
        grid = torch.as_tensor(np.array(grid), dtype=torch.long)

        return image, grid, mask


def __len__(self):
    """ get the length of the dataset by counting the number of receipts in the directory containing the receipts' images """
    return len(self.receipts_images)



