import numpy as np
import cv2
import time
from pdf2image import convert_from_path



def check_extention_file(file_name) :
    """ Check if a file is a pdf or jpg or png or jpeg"""

    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')) :
        return "image"
    elif  file_name.lower().endswith('pdf') :
        return "pdf"
    else :
        return "else"


def convert_pdf_to_images(pdf_path,Dpi=300):
    """ Convert a pdf to images.
    :parameters :
        - pdf_path : path to the pdf.
        _ Dpi      : Dots per inch, which is also pixels per inch of the image to produce from the pdf."""
    pages = convert_from_path(pdf_path, dpi=Dpi)
    return pages


def read_image(file_path):
    """
    read image:
    :parameters :
        -file_path  : path to the file to read.
    """
    try :
        file_extention = check_extention_file(file_path)
        assert file_extention =="image" or file_extention == "pdf"
    except :
        print( " the read file  is not an image nor a pdf. Check the extension of the file")

    else :
        if file_extention =="image" :
            return cv2.imread(file_path)
        else :
            return convert_pdf_to_images(file_path)



def normalize_image(image):
    """
    Normaliser l'image
    """
    return (image / 255.).astype(np.uint8)



def sort_contours_selon_y(contours) :
    """ sort boxes according to the coordinate y. """
    # get coordonnées.
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    # d'abord trier en fonction de y pour avoir les lignes.
    boundingBoxes = sorted(boundingBoxes,key=lambda b:b[1])
    return boundingBoxes

def sort_contours_selon_x_y(contours, line_diff=30):
    """  Detect lines and sort them according to x and y.
    parametres :
        - countours     : contours in image   .
        - line_diff     :  the diff between cordinates y of two boxes to judge is they are in the same line or not.
    return : boundingBoxes for each contours sorted into lines and columns.."""
    boundingBoxes = sort_contours_selon_y(contours)
    out_put = []
    Line = []
    y_old = -100
    for (x, y, w, h) in boundingBoxes:
        # meme ligne.
        if y - y_old < line_diff:
            Line += [(x, y, w, h)]
        # else it's a new line.
        else:
            if Line != []:
                # sort line according to x.
                Line = sorted(Line, key=lambda b: b[0])
                # sauvegarder la ligne
                out_put += [Line]
                # go to new ligne
                Line = []
            # add new box to new line.
            Line += [(x, y, w, h)]
        y_old = y
    return out_put




def get_bounding_boxes(image):
    """
    get boundingBoxes of the boxes in image and sorted according to y and x, means in lines and columns .
    parameters :
        - image  : array  image
    returns  :
        - boundingBoxes  :  four cordinates for each box in the image formed with the horiz and verti lines.
        - horiz_lines_before_completing :  horizontal lines in image.
        - vertic_lines                  :  vertical lines in image
    """
    # if image is RGB put it to graysclae.
    if len(np.shape(image)) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_2 = np.copy(image)

    ## Get contours in this image.
    contours, hierarchy = cv2.findContours(image_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = sort_contours_selon_x_y(contours)
    return boundingBoxes





def draw_box_on_image(boundingBoxes, image, file_name):
    """
    draw the boundings boxes on the image and order them using and index written on the middle of  each bounding box.
    """
    original_image = np.copy(image)
    height, width, _ = np.shape(original_image)
    index = 0
    for line in boundingBoxes:
            for box in line:
                (x, y, w, h) = box
                index += 1

                original_image = write_text_image(image=original_image, text_to_write=str(index),
                                                    position=(x + w // 2, y + h // 2), color=(255, 0, 255, 0),
                                                    fontSize=1.5)
                original_image = cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 5)

    cv2.imwrite(file_name, original_image)



if __name__ == '__main__': 
    
    file_name = "hamza_ouhssaine222.png"
    
    read_image(file_name)
