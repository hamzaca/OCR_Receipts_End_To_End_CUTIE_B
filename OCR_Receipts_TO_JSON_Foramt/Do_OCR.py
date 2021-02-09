import utilis
from PIL import Image
import pyocr
import pyocr.builders



tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
# # The tools are returned in the recommended order of usage
tool = tools[0]
print("Will use tool '%s'" % (tool.get_name()))
# # Ex: Will use tool 'libtesseract'
#
langs = tool.get_available_languages()
print("Available languages: %s" % ", ".join(langs))
lang = langs[0]+ '+'+langs[1]
print("Will use lang '%s'" % (lang))


def get_name_receipt_with_extention(receipt_name):
    index = receipt_name.index(".")
    return receipt_name[0:index]



def ocr_one_image_to_json(receipt_name):

    image = utilis.read_image(receipt_name)
    boundingBoxes = utilis.get_bounding_boxes(image)

    f = open(get_name_receipt_with_extention+".txt", "a")

    list_text_boxes = []
    for index,bbox in enumerate(boundingBoxes):
        (x, y, w, h) = bbox
        bbox_image = image[y:y + h, x:x + w]
        bbox_text = tool.image_to_string(Image.fromarray(new_img),lang=lang,builder=pyocr.builders.TextBuilder())
        dict_this_box = {}
        dict_this_box["id"] = index+1
        dict_this_box["bbox"] =[x,y,x+w,y+h]
        dict_this_box["text"] = bbox_text
        list_text_boxes += [dict_this_box]
        ##TODO :  know whether it's a line that has the total in it.
        if bbox_text.lower()=='total' :

    dict_this_image = {"text_boxes" : list_text_boxes}


    f.write( dict_this_image)



## TODO : DO ocr for all receipts' images. Get the JSON format for all.
## Read all files names in the folder /receipts_images
"""
receipts_names = os.listdir(path="receipts_images")
# order the receipts so  the names start from '1000-receipt.jpng' tell the end.
receipts_names.sort()

for receipt_name in receipts_names :
    image = utilis.read_image(receipt_name)
    boundingBoxes = utilis.get_bounding_boxes(image)




 """

if __name__ == '__main__':

    print(get_name_receipt_with_extention("ha_1000mza_22.jpg"))





