from PIL import Image
import pytesseract
import json
import utilis
import os

def read_data_from_ocr(data):
    rows = data.split('\n')
    nlen = len(rows)
    if nlen == 0:
        return
    doc = []
    c = dict()
    header = rows[0].split('\t')

  
    for i in range(1, nlen):
        r = rows[i].split('\t')
        word = dict()
        word['id'] = i
        word['bbox'] = tuple([int(r[6]), int(r[7]), int(r[6]) + int(r[8]), int(r[7]) + int(r[9])])
        if len(r) < 12:
            word['text'] = ""
        else:
            word['text'] = r[11]
        if word['text']:
            doc.append(word)
    return doc


def receipt_ton_json(receipt_path):
    # A receipt maybe an image or a pdf.
    extension =  utilis.check_extention_file(receipt_path)
    # path of the json file after ocr.
    path_jsons = "/receipts_jsons"
    json_path = os.path.join(path_jsons, utilis.get_name_receipt_without_extention(receipt_path)+".json")
    try :
      assert extension!='else'
    except :
      print(" The file read is not and image or a pdf. Check whether the directory contains files that are not receipts. ")
    else :
      if extension == "image" :
        img = Image.open(receipt_path)
      elif  extension =="pdf" :
          img = Image.fromarray(convert_pdf_to_images(receipt_path))
      doc = pytesseract.image_to_data(img, lang='fra+eng', config='--psm 11')  # custom_config
      doc = read_data_from_ocr(doc)
      new_doc = {}
      new_doc.update({'text_boxes': doc})
      classes = ["DontCare", "TotalTTC"]
      fields = []
      for cl in classes:
          new_field = {"field_name": cl, "value_id": [], "value_text": [], "key_id": [], "key_text": []}
          fields.append(new_field)
      new_doc.update({'fileds': fields})
      new_doc.update({"global_attributes": {"file_id": receipt_path}})
      json_obj = json.dumps(new_doc, indent=4, ensure_ascii=False)
      f = open(json_path, "a")
      f.write(json_obj)
      f.close()


if __name__ == '__main__':
    path_images = "receipts_images"
    for one_receipt_path in os.listdir(path_images):
        print(" OCR to JSON for receipt {} is on the way".format(one_receipt_path))
        receipt_ton_json(receipt_path=os.path.join(path_images, one_receipt_path))
        print(" OCR to JSON for receipt {} is done".format(one_receipt_path))





