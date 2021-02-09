from PIL import Image
import pytesseract
import sys
import json


def read_data_from_ocr(data):
    rows = data.split('\n')
    nlen = len(rows)
    if nlen == 0:
        return
    doc = []
    c = dict()
    header = rows[0].split('\t')

    npage = None  # page#
    nbl = None
    nline = None
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


def main(receipt_path,  json_path):

    with Image.open(receipt_path) as img:

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


if __name__ == "__main__":
    main()