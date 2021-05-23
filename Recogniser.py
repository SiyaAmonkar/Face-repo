import cv2                  # Importing the opencv
import NameFind             # Import Functions
import pyttsx
engine=pyttsx.init()
# import the Haar cascades for face and eye ditection
face_cascade = cv2.CascadeClassifier('C:\\Users\\HP\\Downloads\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\HP\\Downloads\\opencv\\sources\\data\\haarcascades\\haarcascade_eye.xml')
recognise = cv2.createEigenFaceRecognizer(15, 4000)                       # creating EIGEN FACE RECOGNISER
recognise.load("C:\\Python27\\face\\Recogniser\\trainingDataEigan.xml")
recognise = cv2.createFisherFaceRecognizer(5, 600)                        # creating FISHER FACE RECOGNISER
recognise.load("C:\\Python27\\face\\Recogniser\\trainingDataFisher.xml")  # Load the training data from the trainer to recognise the faces

#START THE VIDEO FEED 
cap = cv2.VideoCapture(0)                                               # Camera object
ID = 0
while True:
    ret, img = cap.read()                                               # Read the camera object
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                        # Convert the Camera to gray
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)                 # Detect the faces and store the positions
    for (x, y, w, h) in faces:                                          # Frames  LOCATION X, Y  WIDTH, HEIGHT
        gray_face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))       # The Face is isolated and cropped
   
        eyes = eye_cascade.detectMultiScale(gray_face)
        for (ex, ey, ew, eh) in eyes:
            ID, conf = recognise.predict(gray_face)# Determine the ID of the photo
            NAME = NameFind.ID2Name(ID ,conf)      
            NameFind.DispID(x, y, w, h, NAME, gray)
            engine.say(NAME)
            engine.runAndWait()
    cv2.imshow('Face Recognition System', gray)                         # Show the video
    if cv2.waitKey(1) & 0xFF == ord('q'):                               # Quit if the key is Q
        break
cap.release()
cv2.destroyAllWindows()
