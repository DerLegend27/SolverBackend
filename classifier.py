from keras.preprocessing.image import img_to_array
from keras.models import load_model
import os
import numpy as np
import pickle
import cv2

MODEL = "model.model"
LABELS = "labels.pickle"

def classification():

	# Define Solution
	solution = ""
	
	# Loading the CNN and MLB (multi-label-binarizer)
	print("[INFO] CNN wird geladen...")
	model = load_model(MODEL)
	mlb = pickle.loads(open(LABELS, "rb").read())

	for i in os.listdir("result-images"):
		
		image = cv2.imread("result-images/" + str(i))

		# Pre-process the image
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)

		# Classify the image
		print("[INFO] Image wird classifiziert...")
		proba = model.predict(image)[0]

		print ("Folgendes Symbol wurde erkannt: ",' '.join(mlb.classes_[proba.argmax(axis=-1)]))
		solution = solution + str(mlb.classes_[proba.argmax(axis=-1)])

		os.remove("result-images/" + str(i))

	print(solution)	
	return solution

if __name__ == "__main__":
	classification()
