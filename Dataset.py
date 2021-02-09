from torch.utils.data import Dataset
from gensim import corpora
from matplotlib import cm
from PIL import Image
import os



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
    boxes = get_boxes(self.receipts_json[idx])
    grid, self.dictionary = get_grid(one_receipt_image, self.dictionary)
    mask = get_mask(boxes, w, h)

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



