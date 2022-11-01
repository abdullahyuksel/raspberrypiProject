import pandas as pd
import dlib
import cv2
from sklearn.linear_model import LinearRegression
import time
from gpiozero import Buzzer

buzzer = Buzzer("GPIO17")

dataset = pd.read_csv("dataset.csv")
x = dataset.iloc[:,:3].values
y = dataset.iloc[:,3:].values


lr = LinearRegression()
lr.fit(x,y)


cap = cv2.VideoCapture(1)

detector = dlib.get_frontal_face_detector()
model = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def mid(p1,p2):
    return ( int((p1[0]+p2[0])/2) , int((p1[1]+p2[1])/2)  )

say = 0

while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    faces = detector(frame)

    for face in faces:
        points = model(gray,face)
        points_list = [(p.x,p.y) for p in points.parts()]


        #################################################
        sag_ust = mid( points_list[37] , points_list[38] )
        sag_alt = mid( points_list[41] , points_list[40] )
        sag_mesafe = sag_alt[1] - sag_ust[1]

        cv2.circle(frame,sag_ust,3,(0,0,255),-1)
        cv2.circle(frame,sag_alt,3,(255,0,0),-1)
        ###############################################
        sol_ust = mid( points_list[43] , points_list[44] )
        sol_alt = mid( points_list[47] , points_list[46] )
        sol_mesafe = sol_alt[1] - sol_ust[1]

        cv2.circle(frame,sol_ust,3,(0,0,255),-1)
        cv2.circle(frame,sol_alt,3,(255,0,0),-1)

        burun_mesafe = points_list[30][1] - points_list[27][1]

        pred = lr.predict([[sol_mesafe,sag_mesafe,burun_mesafe]])
        pred_list = []
        for i in pred:
            sol = 0
            sag = 0
            if i[0] > 0.5:
                sol = 1
            if i[1] > 0.5:
                sag = 1
            pred_list.append([sol,sag])

        if pred_list[0][0] == 0 and pred_list[0][1] == 0:
            say += 1
        else:
            say = 0

        if say >= 30:
            buzzer.on()
        else:
            buzzer.off()
        



    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



