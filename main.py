import cv2

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    casc = cv2.CascadeClassifier("haarcascade_frontalface_default (2).xml")
    face_rect = casc.detectMultiScale(gray, 1.1, 9)
    for (x,y,w,h) in face_rect:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Deney", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
