#import the necessary packages
from imutils import paths
import PIL.Image
import dlib
import argparse
import pickle
import cv2
import os
import numpy as np


def face_distance(face_encodings, face_to_compare):
	if len(face_encodings) == 0:
		return np.empty((0))

	return np.linalg.norm(face_encodings - face_to_compare, axis=1)



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--dataset", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

total_index = 0
the_true_index = 0
the_false_index = 0


face_detector = dlib.get_frontal_face_detector()
predictor_68_point_model = "shape_predictor_68_face_landmarks.dat"
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

cnn_face_detection_model = "mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = "dlib_face_recognition_resnet_model_v1.dat"
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())



image_paths = list(paths.list_images(args["dataset"]))

for test_image in image_paths :

	# load the input image and convert it from BGR to RGB
	image = cv2.imread(test_image)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image, then compute the facial embeddings
	# for each face
	print("[INFO] recognizing faces...")
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


	# initialize the list of names for each face detected
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = list(face_distance(data["encodings"], encoding) <= 0.6)

		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number of
			# votes (note: in the event of an unlikely tie Python will
			# select first entry in the dictionary)
			name = max(counts, key=counts.get)

		# update the list of names
		names.append(name)
		for name in names:
			total_index += 1
			if name == "{}".format(test_image.split('\\')[-2]) :
				the_true_index += 1
			else:
				the_false_index += 1

	# loop over the recognized faces
	for (a, name) in zip(boxes, names):
		# draw the predicted face name on the image
		top = a.top()
		right = a.right()
		bottom = a.bottom()
		left = a.left()
		cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

	# save the output image
	cv2.imwrite("tested-images/{}".format(test_image),image)


print("_______________________________________________________________________________")
print("_______________________________________________________________________________")
print("_______________________________________________________________________________")
print("the ratio of true predictions : "+str((the_true_index/total_index)*100)+"%")
print("_______________________________________________________________________________")
print("the ratio of false predictions : "+str((the_false_index/total_index)*100)+"%")
print("_______________________________________________________________________________")
print("_______________________________________________________________________________")
print("_______________________________________________________________________________")
