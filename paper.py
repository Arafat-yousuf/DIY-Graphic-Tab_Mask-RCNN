"""
Mask R-CNN
Train on the Paper dataset and implement warp and threshold.

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 paper.py train --dataset=/path/to/paper/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 paper.py train --dataset=/path/to/paper/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 paper.py train --dataset=/path/to/paper/dataset --weights=imagenet

    # Apply warp and threshold to an image
    python3 paper.py warp --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply warp and threshold to video using the last weights you trained
    python3 paper.py warp --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import glob
import cv2
import time
import datetime
import numpy as np
import skimage.draw
from matplotlib import pyplot as plt
import imutils

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import visualize
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################
class CCA:
    def __init__(self, output_process = False):
        self.output_process = output_process
    def __call__(self, image):
        
#         2nd argument is either 4 or 8, denoting the type of Connected Component Analysis
        (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(image,8, cv2.CV_32S)

        max_area = -1
        max_area_label = -1
        
        if self.output_process:
            print("numlabels -- ",numLabels)
            
        for i in range(1,numLabels):
            temp_area = stats[i, cv2.CC_STAT_AREA]
            
            if self.output_process:
                print(temp_area)
                
            if temp_area > max_area : 
                max_area = temp_area
                max_area_label = i

        res_image = (labels == max_area_label).astype("uint8") * 255

        return res_image
    
class Dilation:
    def __init__(self, kernel_size = 3, iterations = 25, output_process = False):
        self._kernel_size = kernel_size
        self._iterations = iterations
        self.output_process = output_process


    def __call__(self, image):
        
        start = time.time()
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self._kernel_size, self._kernel_size)
        )
        
        dilated = cv2.dilate(image,kernel,iterations = self._iterations )
        
        end = time.time()

        if self.output_process:
            print("After executing Dilation ---" , (end-start))
            
        return dilated


class Closer:
    def __init__(self, kernel_size = 3, iterations = 10, output_process = False):
        self._kernel_size = kernel_size
        self._iterations = iterations
        self.output_process = output_process

    def __call__(self, image):
        
        start = time.time()
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self._kernel_size, self._kernel_size)
        )
        closed = cv2.morphologyEx(
            image, 
            cv2.MORPH_CLOSE, 
            kernel,
            iterations = self._iterations
        )

        end = time.time()
        
        if self.output_process:
            print("After executing Closer ---" , (end-start))
            
        return closed

class OtsuThresholder:
    
    def __init__(self, thresh1 = 0, thresh2 = 255, output_process = False):
        self.output_process = output_process
        self.thresh1 = thresh1
        self.thresh2 = thresh2

    def __call__(self, image):
        
        start = time.time()
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        T_, thresholded1 = cv2.threshold(image, self.thresh1, self.thresh2, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresholded2 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,2)
        
        end = time.time()
        
        if self.output_process:
            print("After executing Otsu thresholder ---" , (end-start))
            
        return thresholded1,thresholded2
    
def hand_remove(img):
    
    otsu_obj = OtsuThresholder(thresh1 = 128, thresh2 = 255, output_process = False)
    close_obj = Closer(iterations = 5,output_process = False)
    dilate_obj = Dilation(iterations = 1,output_process = False)
    cca_obj = CCA(output_process = False)
    
    p,q = otsu_obj(img)
    p = close_obj(p)
    p = cca_obj(~p)
    p = dilate_obj(p)
    p = q | p
    p = dilate_obj(p)
    
    return p


class PaperConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "paper"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + paper

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class PaperDataset(utils.Dataset):

    # def load_paper(self, dataset_dir, subset):
    #     """Load a subset of the Paper dataset.
    #     dataset_dir: Root directory of the dataset.
    #     subset: Subset to load: train or val
    #     """
    #     # Add classes. We have only one class to add.
    #     self.add_class("paper", 1, "paper")

    #     # Train or validation dataset?
    #     assert subset in ["train", "val"]
    #     dataset_dir = os.path.join(dataset_dir, subset)

    #     img_dir = "image/"
    #     txt_dir = "text/"

    #     data_path = os.path.join(dataset_dir, img_dir)
    #     txt_dir = os.path.join(dataset_dir, txt_dir)

    #     # files = glob.glob(data_path + '/*')
    #     files = [os.path.normpath(i) for i in glob.glob(data_path + '/*')]
    #     # print(files)
    #     #files.sort() #We sort the images in alphabetical order to match them to the xml files containing the annotations of the bounding boxes

    #     for f1 in files:
    #         img = cv2.imread(f1)
    #         height, width = img.shape[:2]
    #         # print(height, width)

    #         pp = f1
            
    #         pp = pp.split('\\')
            
    #         pp = pp[8]
            
    #         pp = pp.split('.')
    #         pp = pp[0]
            
    #         img_name = pp + '.jpg'
    #         print(img_name)

    #         p = txt_dir + pp + '.txt'
    #         image_path = data_path + pp + '.jpg'
    #         file1 = open(p, "r")
    #         Fc = file1.read()
    #         Fc = json.loads(Fc)
    #         Fc = np.array(Fc)
    #         Fc = Fc.flatten()
    #         Fc = np.int32(Fc)
    #         # print(Fc)

    #         self.add_image(
    #             "paper",
    #             image_id=img_name,  # use file name as a unique image id
    #             path=image_path,
    #             width=width, height=height,
    #             polygons=Fc)

    def load_pp(self, img_name, image_path, width, height, Fc):
        """Load a subset of the Paper dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("paper", 1, "paper")

        self.add_image(
            "paper",
            image_id=img_name,  # use file name as a unique image id
            path=image_path,
            width=width, height=height,
            polygons=Fc)

    

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a paper dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "paper":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        # print(info)
        mask = np.zeros([info["height"], info["width"],  1], dtype=np.uint8)
        ycord = [info["polygons"][0],info["polygons"][2],info["polygons"][4],info["polygons"][6]]
        xcord = [info["polygons"][1],info["polygons"][3],info["polygons"][5],info["polygons"][7]]
        print(xcord)
        rr, cc = skimage.draw.polygon(xcord, ycord)
        mask[rr, cc, 0] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "paper":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = PaperDataset()
    dataset_train.load_paper(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PaperDataset()
    dataset_val.load_paper(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

# def gd1(pt,lst):
#     pt = pt / 2
#     lt =[]
#     rt =[]
#     for i in range(4):
#         if lst[i][0]<=pt:
#             lt.append([lst[i][0],lst[i][1]])
#         else :
#             rt.append([lst[i][0],lst[i][1]])
#     return lt,rt

def orientation(o,a,b):
    return (b[0][1]-a[0][1])*(a[0][1]-o[0][0]) - (a[0][1]-o[0][1])*(b[0][0]-a[0][0])
def dist(a,b):
    return (a[0][0]-b[0][0])*(a[0][0]-b[0][0]) + (a[1][0]-b[0][1])*(a[0][1]-b[0][1])
def comp(a,b,po):
    ori = orientation(po,a,b)
    if ori==0 :
        return dist(po,b)>=dist(po,a)
    return ori>0 
def orient(pts):
    global po
    if pts.shape[0]!=4:
        print("need exactly 4 points")
        return pts;
    ind = 0
    for i in range(4):
        if pts[i][0][1]<pts[ind][0][1] or (pts[i][0][1]==pts[ind][0][1] and pts[i][0][0]<pts[ind][0][0]):
            ind =i
    pts[[0,ind]]= pts[[ind,0]]
    for i in range(1,4):
        for j in range (i+1,4):
            if comp(pts[i],pts[j],pts[0]):
                pts[[i,j]]=pts[[j,i]]
    return pts

# def gd(lst,pt):
#     lt =[]
#     rt =[]
#     pt = pt / 2 + 50
#     rect = np.zeros((4, 2), dtype = "float32")
#     for i in range(4):
#         if lst[i][0]<=pt:
#             lt.append([lst[i][0],lst[i][1]])
#         else :
#             rt.append([lst[i][0],lst[i][1]])
#     # print(lt)
#     # print(rt)
#     rect[3] = lt[0]
#     rect[2] = lt[1]
#     rect[0] = rt[0]
#     rect[1] = rt[1]
#     if lt[0][1]>lt[1][1]:
#         rect[3] =lt[1]
#         rect[2] =lt[0]
#     if rt[0][1]>rt[1][1]:
#         rect[0] =rt[1]
#         rect[1] =rt[0]
    
#     return rect

def gd(lst):
    rect = np.zeros((4, 2), dtype = "float32")
    lt =[]
    rt =[]
    for i in range(4):
        for j in range(i+1,4):
            if(lst[i][0]>lst[j][0]):
                lst[[i,j]]= lst[[j,i]]
    lt.append(lst[0])
    lt.append(lst[1])
    rt.append(lst[2])
    rt.append(lst[3])
    rect[3] = lt[0] # bl
    rect[2] = lt[1] # br
    rect[0] = rt[0] # tl
    rect[1] = rt[1] # tr
    if lt[0][1]>lt[1][1]:
        rect[3] =lt[1]
        rect[2] =lt[0]
    if rt[0][1]>rt[1][1]:
        rect[0] =rt[1]
        rect[1] =rt[0]
    return rect


def order_points(pts,width):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    width = width / 2
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[2]
    rect[2] = pts[0]
    # rect[0] = pts[np.argmin(s)]
    # rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[1]
    rect[3] = pts[3]
    # rect[1] = pts[np.argmin(diff)]
    # rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
    # print("pts---",pts)
    # rect = order_points(pts,width)
    rect = gd(pts)
    # print("rect---",rect)
    (tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # print("warped shape--",warped.shape)
	# return the warped image
    return warped

# def generate_warp(image, mask):
#     """Apply warp and threshold effect.
#     image: RGB image [height, width, 3]
#     mask: instance segmentation mask [height, width, instance count]

#     Returns result image.
#     """
#     # Make a grayscale copy of the image. The grayscale copy still
#     # has 3 RGB channels, though.
#     gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
#     # Copy color pixels from the original color image where mask is set
#     if mask.shape[-1] > 0:
#         # We're treating all instances as one, so collapse the mask into one layer
#         mask = (np.sum(mask, -1, keepdims=True) >= 1)
#         warp = np.where(mask, image, gray).astype(np.uint8)
#     else:
#         warp = gray.astype(np.uint8)
#     return warp







# def detect_and_warp(model, image_path=None, video_path=None):
#     assert image_path or video_path

#     class_names = ['BG', 'paper']

#     # Image or video?
#     if image_path:
#         # Run model detection and generate the warp and threshold effect
#         print("Running on {}".format(args.image))
#         # Read image
#         image = skimage.io.imread(args.image)
#         # Detect objects
#         r = model.detect([image], verbose=1)[0]
#        # warp and threshold
#         # warp = generate_warp(image, r['masks'])
#         visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                                     class_names, r['scores'], making_image=True)
#         file_name = 'warp.png'
#         # Save output
#         # file_name = "warp_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
#         # save_file_name = os.path.join(out_dir, file_name)
#         # skimage.io.imsave(save_file_name, warp)
#     elif video_path:
#         import cv2
#         # Video capture
#         vcapture = cv2.VideoCapture(video_path)
#         # width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
#         # height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         width = 1280
#         height = 720
#         # fps = vcapture.get(cv2.CAP_PROP_FPS)
#         fps = 5
#         # Define codec and create video writer
#         file_name = "warp_{:%Y%m%dT%H%M%S}.wmv".format(datetime.datetime.now())
#         vwriter = cv2.VideoWriter(file_name,
#                                   cv2.VideoWriter_fourcc(*'MJPG'),
#                                   fps, (width, height))

#         count = 0
#         success = True
#         #For video, we wish classes keep the same mask in frames, generate colors for masks
#         colors = visualize.random_colors(len(class_names))
#         while success:
#             print("frame: ", count)
#             # Read next image
#             plt.clf()
#             plt.close()
#             success, image = vcapture.read()
#             if success and count % 5 == 0:
#                 # OpenCV returns images as BGR, convert to RGB
#                 image = image[..., ::-1]
#                 # Detect objects
#                 r = model.detect([image], verbose=0)[0]
#                 # warp and threshold
#                 # warp = generate_warp(image, r['masks'])

#                 warp = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                                                     class_names, r['scores'], making_video=True)
#                 # Add image to video writer
#                 vwriter.write(warp)
#             count += 1
#         vwriter.release()
#     print("Saved to ", file_name)



def generate_warp(image, mask):
    """Apply warp and threshold effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        mask1 = ~mask
        warp = np.where(mask, image, 0).astype(np.uint8)
        warp = np.where(mask1, warp, 255).astype(np.uint8)
    else:
        warp = gray.astype(np.uint8)
    return warp


# def detect_and_warp(model, image_path=None, video_path=None):
#     assert image_path or video_path

#     # Image or video?
#     if image_path:
#         # Run model detection and generate the warp and threshold effect
#         print("Running on {}".format(args.image))
#         # Read image
#         image = skimage.io.imread(args.image)
#         # Detect objects
#         r = model.detect([image], verbose=1)[0]
#         # warp and threshold
#         warp = generate_warp(image, r['masks'])
#         # Save output
#         file_name = "warp_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
#         skimage.io.imsave(file_name, warp)
#     elif video_path:
#         import cv2
#         # Video capture
#         vcapture = cv2.VideoCapture(video_path)
#         width1 = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height1 = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         width = 500
#         height = 888
#         fps = vcapture.get(cv2.CAP_PROP_FPS)
#         # fps = 5
#         # Define codec and create video writer
#         file_name = "warp_{:%Y%m%dT%H%M%S}.mp4".format(datetime.datetime.now())
#         vwriter = cv2.VideoWriter(file_name,
#                                   cv2.VideoWriter_fourcc(*'X264'),
#                                   fps, (width, height))

#         count = 0
#         success = True
#         sm1 = [0, 0]
#         succ = False
#         while success:
#             print("frame: ", count)
#             # Read next image
#             success, image = vcapture.read()
#             orig = image
#             if success:
#                 # OpenCV returns images as BGR, convert to RGB
#                 image = image[..., ::-1]
#                 # Detect objects
#                 if count % 15 ==0:
#                     r = model.detect([image], verbose=0)[0]
#                 # warp and threshold
#                 warp = generate_warp(image, r['masks'])
                
#                 # RGB -> BGR to save image to video
#                 warp = warp[..., ::-1]
#                 # print(warp.shape)
#                 gry = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
#                 kernel = np.ones((8,8), np.uint8) 
#                 warp = cv2.dilate(gry,kernel)
#                 gry = cv2.GaussianBlur(gry, (5, 5), 0)
#                 edged = cv2.Canny(gry, 75, 200)
#                 # print(edged.shape)
                
#                 # TEST 01
#                 cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#                 cnts = imutils.grab_contours(cnts)
#                 cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
#                 # loop over the contours
#                 for c in cnts:
#                     peri = cv2.arcLength(c, True)
#                     approx = cv2.approxPolyDP(c, 0.02 * peri, True)
# 	                # if our approximated contour has four points, then we
# 	                # can assume that we have found our screen
#                     if len(approx) == 4:
#                         screenCnt = approx
#                         succ = True
#                         break
#                 edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

#                 if succ:
#                     cv2.drawContours(edged, [screenCnt], -1, (0, 255, 0), 2)
#                     # print("edged shape--",edged.shape)
#                     # edged = cv2.resize(edged, (width,height), interpolation = cv2.INTER_AREA)
#                     # TEST 01 END


#                     # edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
#                     # Add image to video writer
#                     # screenCnt1 = screenCnt
#                     print("screenCnt---",screenCnt)
#                     sm = sum(screenCnt)
#                     sm = sm[0]
#                     print("sum----",sm)
#                     # screenCnt = orient(screenCnt)
#                     # print("Here lies Bellman--",screenCnt)
#                     if (((sm[0]<sm1[0]-50) or (sm[0] > sm1[0] + 50)) or ((sm[1] < sm1[1]-50) or (sm[1] > sm1[1] + 50))):   
#                         screenCnt1 = screenCnt
#                         sm1 = sm
#                         print("hereeee")
#                     warped = four_point_transform(orig, screenCnt1.reshape(4, 2))
#                     print("sum1---",sm1) 
#                     print("screenCnt1---",screenCnt1) 
#                     # convert the warped image to grayscale, then threshold it
#                     # to give it that 'black and white' paper effect
#                     # warped = cv2.cvtColor(warped)
#                     # T = threshold_local(warped, 11, offset = 10, method = "gaussian")
#                     # warped = (warped > T).astype("uint8") * 255
#                     # print("warped111 shape--",warped.shape)
#                     warped = cv2.resize(warped, (width,height), interpolation = cv2.INTER_AREA)
#                     print("warpedres shape--",warped.shape)
#                     vwriter.write(warped)
#             count += 1
#         vwriter.release()
#     print("Saved to ", file_name)

def detect_and_warp(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the warp and threshold effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # warp and threshold
        warp = generate_warp(image, r['masks'])
        # Save output
        file_name = "warp_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, warp)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width1 = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height1 = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = 500
        height = 888
        fps = vcapture.get(cv2.CAP_PROP_FPS)
        # fps = 5
        # Define codec and create video writer
        file_name = "warp_{:%Y%m%dT%H%M%S}.mp4".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'X264'),
                                  fps, (width, height))

        count = 0
        success = True
        sm1 = [0, 0]
        succ = False
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            orig = image
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                if count % 15 ==0:
                    r = model.detect([image], verbose=0)[0]
                # warp and threshold
                warp = generate_warp(image, r['masks'])
                
                # RGB -> BGR to save image to video
                warp = warp[..., ::-1]
                print(warp.shape)
                gry = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((8,8), np.uint8) 
                warp = cv2.dilate(gry,kernel)
                gry = cv2.GaussianBlur(gry, (5, 5), 0)
                edged = cv2.Canny(gry, 75, 200)
                print(edged.shape)
                
                # TEST 01
                cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                cnts = imutils.grab_contours(cnts)
                cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
                # loop over the contours
                for c in cnts:
                    peri = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	                # if our approximated contour has four points, then we
	                # can assume that we have found our screen
                    if len(approx) == 4:
                        screenCnt = approx
                        succ = True
                        break
                edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

                if succ:
                    cv2.drawContours(edged, [screenCnt], -1, (0, 255, 0), 2)
                    # print("edged shape--",edged.shape)
                    # edged = cv2.resize(edged, (width,height), interpolation = cv2.INTER_AREA)
                    # TEST 01 END


                    # edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
                    # Add image to video writer
                    # screenCnt1 = screenCnt
                    # print("screenCnt---",screenCnt)
                    sm = sum(screenCnt)
                    sm = sm[0]
                    # print("sum----",sm)
                    # screenCnt = orient(screenCnt)
                    # print("Here lies Bellman--",screenCnt)
                    if (((sm[0]<sm1[0]-50) or (sm[0] > sm1[0] + 50)) or ((sm[1] < sm1[1]-50) or (sm[1] > sm1[1] + 50))):   
                        screenCnt1 = screenCnt
                        sm1 = sm
                    warped = four_point_transform(orig, screenCnt1.reshape(4, 2))
                    # print("sum1---",sm1) 
                    # print("screenCnt1---",screenCnt1) 
                    # convert the warped image to grayscale, then threshold it
                    # to give it that 'black and white' paper effect
                    # warped = cv2.cvtColor(warped)
                    # T = threshold_local(warped, 11, offset = 10, method = "gaussian")
                    # warped = (warped > T).astype("uint8") * 255
                    # print("warped111 shape--",warped.shape)
                    warped = cv2.resize(warped, (width,height), interpolation = cv2.INTER_AREA)
                    # print("warpedres shape--",warped.shape)
                    res = hand_remove(warped)
                    vwriter.write(res)
            count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    os.makedirs('RESULTS')
    submit_dir = os.path.join(os.getcwd(), "RESULTS/")
    # Read dataset
    dataset = PaperDataset()
    dataset.load_VIA(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        canvas = visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'], detect=True)
            # show_bbox=False, show_mask=False,
            # title="Predictions",
            # detect=True)
        canvas.print_figure("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"][:-4]))
    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect papers.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'warp'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/paper/dataset/",
                        help='Directory of the Paper dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the warp and threshold on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the warp and threshold on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "warp":
        assert args.image or args.video,\
               "Provide --image or --video to apply warp and threshold"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = PaperConfig()
    else:
        class InferenceConfig(PaperConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "warp":
        detect_and_warp(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'warp'".format(args.command))
