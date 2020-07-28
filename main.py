import dateutil.parser as dparser
import csv
import re
import argparse
import pytesseract
import json
import os.path
from utils import visualization_utils as vis_util
from utils import label_map_util
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import sys
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
sys.path.append("..")
CWD_PATH = os.getcwd()



def image_extract(image_path, output_path):
    
    MODEL_NAME = 'model'

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'labelmap.pbtxt')

    # Path to image
    PATH_TO_IMAGE = os.path.join(CWD_PATH, image_path)

    # Number of classes the object detector can identify
    NUM_CLASSES = 1

    # Load the label map.
    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `king`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    image, array_coord = vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.60)

    ymin, xmin, ymax, xmax = array_coord

    shape = np.shape(image)
    im_width, im_height = shape[1], shape[0]
    (left, right, top, bottom) = (xmin * im_width,
                                  xmax * im_width, ymin * im_height, ymax * im_height)

    # Using Image to crop and save the extracted copied image
    im = Image.open(image_path)
    im.crop((left, top, right, bottom)).save(output_path, quality=95)




def adhaar_text_extractor(output_path, json_output_folder):

    img = cv2.imread(output_path, 0)
    th, threshed = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string((threshed))
    #print(list(filter(lambda x: ord(x) < 128, text)))

    # Initializing data variable
    name = None
    gender = None
    ayear = None
    uid = None
    yearline = []
    genline = []
    nameline = []
    text1 = []
    text2 = []
    genderStr = '(Female|Male|emale|male|ale|FEMALE|MALE|EMALE)$'

    # Searching for Year of Birth
    lines = text

    for wordlist in lines.split('\n'):
        xx = wordlist.split()
        if [w for w in xx if re.search('(Year|Birth|irth|YoB|YOB:|DOB:|DOB|:)$', w)]:
            yearline = wordlist
            break
        else:
            text1.append(wordlist)
    try:
        text2 = text.split(yearline, 1)[1]
    except Exception:
        pass

    try:
        yearline = re.split('Year|Birth|irth|YoB|YOB:|DOB:|DOB', yearline)[1:]
        yearline = ''.join(str(e) for e in yearline)
        if yearline:
            ayear = dparser.parse(yearline, fuzzy=True).year
    except Exception:
        pass

    # Searching for Gender
    try:
        for wordlist in lines.split('\n'):
            xx = wordlist.split()
            if [w for w in xx if re.search(genderStr, w)]:
                genline = wordlist
                break

        if 'Female' in genline or 'FEMALE' in genline:
            gender = "Female"
        elif 'Male' in genline or 'MALE' in genline:
            gender = "Male"

        text2 = text.split(genline, 1)[1]
    except Exception:
        pass

    # Printing the name of the user
    for i in range(len(text1)):
        if len(text1[i]) > 5:
            flag = False
            for j in text1[i].split(' '):
                if j[0].isupper() == False or j[1].isupper == True:
                    flag = True
            if flag == False:
                break
    #print("Print Name: ", text1[i])
    name_new = text1[i]  # Extracting name from the image


    # Searching for UID
    uid = []  # Empty list for uid
    try:
        newlist = []
        for xx in text2.split('\n'):
            newlist.append(xx)
        for i in range(len(newlist)):
            if len(newlist[i]) >= 12 and len(newlist[i]) <= 14:
                break
        #print("Extracted UID:", newlist[i])
        uid_new = newlist[i]  # New vid from the image
        for no in uid_new:
            #        print("Uid: ", no)  # To print individual uid elements
            if re.match("^[0-9 ]+$", no):
                uid.append(no)

    except Exception:
        pass

    # Making tuples of data
    data = {}
    data['Name'] = name_new
    data['Gender'] = gender
    data['Birth date'] = yearline
    if len(list(uid)) > 1:
        data['Uid'] = "".join(uid)  # Storing uid into data['Uid']
    else:
        data['Uid'] = None

    # Writing data into JSON
    fName = os.path.join(CWD_PATH, json_output_folder, os.path.basename(output_path).split('.')[0] + '.json')
    with open(fName, 'w') as fp:
        json.dump(data, fp)

    # Reading data back JSON
    with open(fName, 'r') as f:
        ndata = json.load(f)

    print("\n-----------------------------\n")
    print(ndata['Name'])
    print("-------------------------------")
    print(ndata['Gender'])
    print("-------------------------------")
    print(ndata['Birth date'])
    print("-------------------------------")
    print(ndata['Uid'])
    print("-------------------------------")




def pan_text_extractor(output_path, json_output_folder):

    img = cv2.imread(output_path, 0)
    threshed = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 67, 27)

    text = pytesseract.image_to_string((threshed))
    text = list(filter(lambda x: ord(x) < 128, text))

    name = None
    fname = None
    dob = None
    pan = None
    nameline = []
    yearline = []
    panline = []
    text0 = []
    text1 = []
    text2 = []
    govRE_str = '(GOVERNMENT|OVERNMENT|VERNMENT|DEPARTMENT|EPARTMENT\
                 |PARTMENT|ARTMENT|INDIA|NDIA|INCOME|INCOMETAX|TAX)$'
    numRE_str = '(Number|umber|Account|ccount|count|Permanent|\
                 ermanent|manent|PERMANENT|ACCOUNT|NUMBER)$'

    s = []
    for i in text:
        if i == '\n' or i == '':
            text1.append(''.join(x for x in s))
            s = []
        else:
            s.append(i)
    text1.append(''.join(x for x in s))
    #print(text1)

    lineno = 0

    for wordline in reversed(text1):
        xx = wordline.split()
        if ([w for w in xx if re.search(govRE_str, w)]):
            lineno = text1.index(wordline)
            break

    text0 = text1[lineno + 1:]
    #print(text0)

    # Printing the name of the user
    for i in range(len(text0)):
        if len(text0[i]) > 5 and bool(re.search(r'\d', text0[i])) == False:
            w = []
            words = text0[i].split()
            for j in words:
                if len(j) > 3 or j.isupper():
                    w.append(j)
            w = ' '.join(w)
            if len(w) > 5 and w.isupper():
                nameline.append(w)

    try:
        name = nameline[0]
        fname = nameline[1]
    except Exception as ex:
        pass

    try:
        for i in range(len(text0)):
            xx = text0[i].split()
            if [w for w in xx if re.search('(Date|/Date|Dateof|ateof|Birth|irth|DateofBirth|/Dateof)$', w)]:
                dob = text0[i + 1]
                break
            for j in xx:
                obj = re.search(r'(\d+/\d+/\d+)', j)
                if obj:
                    # print(obj)
                    dob = obj.group()
                    break
                else:
                    obj = re.search(r'(\d+-\d+-\d+)', j)
                    if obj:
                        # print(obj)
                        dob = obj.group()
                        break
        # print(dob)
    except Exception:
        pass

    try:
        for item in text1:
            # print(item)
            if item not in nameline and item not in yearline and len(item) != 0:
                if re.search('[a-zA-Z]', item) or re.search(r'\d', item):
                    panline.append(item)
        # print(panline)
        for wordline in panline:
            xx = wordline.split()
            if ([w for w in xx if re.search(numRE_str, w)]):
                pan_all_list = panline[panline.index(wordline) + 1:]
                # print(pan_all_list)
                break

        flag = False
        for pan_all in pan_all_list:
            # count = 0
            # print(pan_all)
            if len(pan_all) == 10 and pan_all.isupper():
                pan = pan_all
                flag = True

            if len(pan_all) > 10:
                for p in pan_all.split(' '):
                    if len(p) == 10 and p.isupper():
                        pan = p
                        flag = True
                        break
            if flag:
                break
        # print(pan)

    except Exception as ex:
        pass

    # Making tuples of data
    data = {}
    data['Name'] = name
    data['Father Name'] = fname
    data['Date of Birth'] = dob
    data['PAN'] = pan

    # Writing data into JSON
    fName = os.path.join(CWD_PATH, json_output_folder, os.path.basename(output_path).split('.')[0] + '.json')
    with open(fName, 'w') as fp:
        json.dump(data, fp)

    # Reading data back JSON
    with open(fName, 'r') as f:
        ndata = json.load(f)

    print("-------------------------------")
    print(ndata['Name'])
    print("-------------------------------")
    print(ndata['Father Name'])
    print("-------------------------------")
    print(ndata['Date of Birth'])
    print("-------------------------------")
    print(ndata['PAN'])
    print("-------------------------------")




def Main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--card', type=str, default='adhaar', const='adhaar', nargs='?', metavar='', help='Type of card you want to detect(adhaar/pan)')
    parser.add_argument(
        '--image_path', type=str, default='test_images/image_a1.jpeg', metavar='', const='test_images/image_a1.jpeg', nargs='?', help='Image used for Text Extraction')
    parser.add_argument(
        '--output_path', type=str, default='test_images/output/output_a1.jpeg', metavar='', const='test_images/output/output_a1.jpeg', nargs='?', help='To store the Cropped Card from the image')
    parser.add_argument(
        '--json_folder', type=str, default='result', const='result', nargs='?', metavar='', help='To store the json output file')

    args = parser.parse_args()

    image_extract(args.image_path, args.output_path)

    if args.card == 'pan':
        pan_text_extractor(args.output_path, args.json_folder)

    elif args.card == 'adhaar':
        adhaar_text_extractor(args.output_path, args.json_folder)


if __name__ == '__main__':
    Main()