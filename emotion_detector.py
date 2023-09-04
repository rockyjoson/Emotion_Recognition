import cv2
import numpy as np
import tensorflow as tf

emotion_dict = {0:"Angry",1:"Fearful",2:"Happy",3:"Neutral",4:"Sad",5:"Surprised"}

emotion_model = tf.keras.models.load_model(r'D:\DL Projects\Projects\Emotion Detection\emotion_model.h5')

# start the webcam feed
#video = cv2.VideoCapture(0)

video=cv2.VideoCapture(r'D:\DL Projects\Projects\Emotion Detection\test_images_videos\different_emotions.mp4')
while True:
    suc,frame=video.read()
    if not suc:
        break
    face_detector=cv2.CascadeClassifier(r'D:\DL Projects\Projects\Emotion Detection\haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    faces=face_detector.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,120),2)
        face = gray[y:y+h,x:x+w]
        resized_img=np.expand_dims(cv2.resize(face,(64, 64)),axis=0) # (1, 64, 64)
        resized_img = np.expand_dims(resized_img, axis=-1)           # (1, 64, 64, 1)

        # predict the emotions
        emotion_prediction = emotion_model.predict(resized_img)
        max_index = int(np.argmax(emotion_prediction))
        cv2.putText(frame,emotion_dict[max_index],(x+5, y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,0),2)
        
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()