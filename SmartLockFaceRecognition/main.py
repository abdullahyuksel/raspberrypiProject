import pandas as pd
import cv2
import dlib
import face_recognition
from sklearn.linear_model import LinearRegression



image_A = face_recognition.load_image_file("abdullah.PNG")
image_A_encoding = face_recognition.face_encodings(image_A)[0]

image_B = face_recognition.load_image_file("asaf.jpeg")
image_B_encoding = face_recognition.face_encodings(image_B)[0]


detector = dlib.get_frontal_face_detector()
model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

dataset = pd.read_csv("dataset.csv")

x = dataset.iloc[:,:3].values
y = dataset.iloc[:,3:].values

lr = LinearRegression()
lr.fit(x,y)

def mid(p1,p2):
    return ( int((p1[0]+p2[0])/2) , int((p1[1]+p2[1]) /2 )  )

cap = cv2.VideoCapture(0)
say = 0
while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face_locations = []
    # yuz tanma için istenilen format ve hızlandırma
    faces = detector(frame)
    for face in faces:
        x = face.left()
        y = face.top() 
        w = face.right()
        h = face.bottom()
        face_locations.append((y,w,h,x))
    
    #göz durumu
    for face in faces:
        points = model(gray,face)
        point_list = [(p.x,p.y) for p in points.parts() ]

        # sag goz için
        sag_ust = mid(point_list[37],point_list[38])
        sag_alt = mid(point_list[41],point_list[40])
        # göz görselleştirme
        cv2.circle(frame,sag_ust,3,(0,0,255),-1)
        cv2.circle(frame,sag_alt,3,(0,0,255),-1)
        sag_goz_mesafe = sag_alt[1] - sag_ust[1]

        # sol göz için
        sol_ust = mid(point_list[43],point_list[44])
        sol_alt = mid(point_list[47],point_list[46])
        # göz görselleştirme
        cv2.circle(frame,sol_ust,3,(0,0,255),-1)
        cv2.circle(frame,sol_alt,3,(0,0,255),-1)
        sol_goz_mesafe = sol_alt[1]-sol_ust[1]

        burun_mesafe = point_list[30][1]- point_list[27][1]

        pred = lr.predict([[sol_goz_mesafe,sag_goz_mesafe,burun_mesafe]])
        pred_list = []
        for p in pred:
            sol = 0
            sag = 0
            if p[0] >0.5:
                sol= 1
            else:
                sol = 0
            if p[1] > 0.5:
                sag = 1
            else:
                sag = 0
            pred_list.append([sol,sag])

        if pred_list[0][0] == 0 and pred_list[0][1] == 0:
            say += 1

    # yuz tanıma işlemi
    faces_encodings = face_recognition.face_encodings(frame,face_locations)

    for face in faces_encodigs:
        y,w,h,x = face_locations[i]
        i+=1
        resultA = face_recognition.compare_faces([image_A_encoding],face)
        resultB = face_recognition.compare_faces([image_B_encoding],face)
        if resultA[0] == True:
            cv2.rectangle(frame,(x,y),(w,h),(0,0,255),2)
            cv2.rectangle(frame,(x,h),(w,h+30),(0,0,255),-1)
            cv2.putText(frame,"ABDULLAH",(x,h+25),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
            if say == 3:
                print("kapı açılıyor")
                say = 0
        elif resultB[0] == True:
            cv2.rectangle(frame,(x,y),(w,h),(0,0,255),2)
            cv2.rectangle(frame,(x,h),(w,h+30),(0,0,255),-1)
            cv2.putText(frame,"ASAF",(x,h+25),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
            if say == 3:
                print("kapı açılıyor")
                say = 0
        else:
            cv2.rectangle(frame,(x,y),(w,h),(0,0,255),2)
            cv2.rectangle(frame,(x,h),(w,h+30),(0,0,255),-1)
            cv2.putText(frame,"TANIMSIZ",(x,h+25),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
            if say == 3:
                print("yüzünüz tanınmadıı için kapı açılmıyor")
                say = 0




    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()