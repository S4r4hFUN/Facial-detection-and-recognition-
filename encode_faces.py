# import the necessary packages
from imutils import paths
import PIL.Image
import dlib
import argparse
import pickle
import cv2
import os
import numpy as np




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


face_detector = dlib.get_frontal_face_detector()
predictor_68_point_model = "shape_predictor_68_face_landmarks.dat"
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

cnn_face_detection_model = "mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = "dlib_face_recognition_resnet_model_v1.dat"
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = []
	if args["detection_method"] == "cnn":
		raw_face_locations = cnn_face_detector(rgb, 1)
		for face in raw_face_locations :
			rect_to_css = face.rect.top(), face.rect.right(), face.rect.bottom(), face.rect.left()
			boxes.append((max(rect_to_css[0], 0), min(rect_to_css[1], rgb.shape[1]), min(rect_to_css[2], rgb.shape[0]), max(rect_to_css[3], 0)))
	else :
		raw_face_locations = face_detector(rgb, 1)
		for face in raw_face_locations :
			rect_to_css = face.top(), face.right(), face.bottom(), face.left()
			boxes.append((max(rect_to_css[0], 0), min(rect_to_css[1], rgb.shape[1]), min(rect_to_css[2], rgb.shape[0]), max(rect_to_css[3], 0)))





	# compute the facial embedding for the face
	boxes = [dlib.rectangle(box[3], box[0], box[1], box[2]) for box in boxes]
	pose_predictor = pose_predictor_68_point
	raw_landmarks = [pose_predictor(rgb, box) for box in boxes]
	encodings = [np.array(face_encoder.compute_face_descriptor(rgb, raw_landmark_set,1)) for raw_landmark_set in raw_landmarks]





	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
