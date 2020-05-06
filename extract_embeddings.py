# Kütüphaneler
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

# Yüz bulma modelinin diskten yüklenmesi
print("[INFO] loading face detector...")
protoPath = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
modelPath = os.path.sep.join(["face_detection_model",
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Özellik çıkarımı yapan modelin yüklenmesi
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# Veri setindeki resimlerin dosya yollarının toplanması
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images("veriseti"))

# Yüz özelliklerinin ve ait olduğu kişierin isimlerin olduğu listeleri
knownEmbeddings = []
knownNames = []

# Veri setindeki toplam kişi sayısı
total = 0
a= 0
for (i, imagePath) in enumerate(imagePaths):
	a=a+1
	# Dosya yollarından kişi ismlerinin çıkarımı
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	# Resimleri yükledikten sonra genişliğinim 600 yapılması
	# Resimlerin boyutlarının elde edilmesi
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# Görüntüden Blob elde edilmesi
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# BLob'un yüz bulma modeline input olarak verilmesi
	# Görüntüdeki yüzlerin elde edilmesi
	detector.setInput(imageBlob)
	detections = detector.forward()

	# Yüz bulunup bulunmadağının kontrol edilmesi
	if len(detections) > 0:
		# we're making the assumption that each image has only ONE
		# face, so find the bounding box with the largest probability
		i = np.argmax(detections[0, 0, :, 2])#bulunan yüzlerden maks olanın indexi tutuyor
		confidence = detections[0, 0, i, 2]#bulunan yüzün oranını alıyor

			# En büyük olasılıklı tespitin aynı zamanda minimum olasılık testimiz anlamına geldiğinden olmak
		#  Zayıf tespitlerin filtrelenmesini sağlamak
		if confidence > 0.5:
			# Yüz çerçeveleyecek kutunun koordinatlarının bulunması
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Yüz görüntüsünün elde edilmesi
			face = image[startY:endY, startX:endX]
			# Yüz boyutlarının elde edilmesi
			(fH, fW) = face.shape[:2]


			# Yüz boyutlarının yeterli olup olmadığının kontrol edilmesi
			if fW < 20 or fH < 20:
				continue


			# ELde edilen yüzün görüntüsünden bir blob inşa edilmesi
			# Blob 'u modele vererek daha sonra 128 boyutlu vektör elde edilmesi
			# Yüzün ölçülmes,
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)

			embedder.setInput(faceBlob)
			vec = embedder.forward()

			#Yüzün sahibinin ve yüzün 128 boyutlu vektörünün tanımlı listelere eklenmesi
			knownNames.append(name)
			knownEmbeddings.append(vec.flatten())
			total += 1

# Yüz sahiplerinin ve yüzlerden çıkarılan özelliklerin diske kaydedilmesi
print("[INFO] serializing {} encodings...".format(total))
print(total)
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open("output/embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()