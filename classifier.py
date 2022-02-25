from keras.preprocessing.image import img_to_array
from keras.models import load_model
import os
from matplotlib.pyplot import contour
import numpy as np
import pickle
import cv2
import natsort

MODEL = "model.model"
LABELS = "labels.pickle"

def classification():

	# Define Solution
	solution = ""
	
	# Loading the CNN and MLB (multi-label-binarizer)
	print("[INFO] CNN wird geladen...")
	model = load_model(MODEL)
	#mlb = pickle.loads(open(LABELS, "rb").read())
	#mlb = pickle.loads(LABELS.read_bytes())

	with open(LABELS, "rb") as ifile:
		mlb = pickle.load(ifile)

	files = os.listdir("result-images/")

	temp_solution = ""

	for i in natsort.natsorted(files,reverse=False):
		
		image = cv2.imread("result-images/" + str(i))
		print(i)

		# Pre-process the image
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)

		# Classify the image
		print("[INFO] Image wird classifiziert...")
		proba = model.predict(image)[0]
		max_proba = mlb.classes_[proba.argmax(axis=-1)]
		
		# print all possibilities
		#for (label, p) in zip(mlb.classes_, proba):
		#	print("{}: {:.2f}%".format(label, p * 100))

		if max_proba == "-" and temp_solution == "-":
			solution = solution[:-1]
			max_proba = "="

		temp_solution = max_proba

		if max_proba == "T":
			max_proba = "x"

		print ("Folgendes Symbol wurde erkannt: ",' '.join(max_proba))

		if "exp" in i:
			solution = solution + "^" + str(max_proba)
			print(solution)
			os.remove("result-images/" + str(i))
			continue

		solution = solution + str(max_proba)
		print(solution)
		os.remove("result-images/" + str(i))

	if solution.find("=") >= 0:
		solution = solution[solution.find("=")+1:]

	print(solution)	
	return solution

if __name__ == "__main__":
	classification()
