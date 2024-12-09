from ultralytics import YOLO
import cv2

model = YOLO('Models/yolo11n.pt')
video_path = 0

cap = cv2.VideoCapture(video_path)

while True:
    success, frame = cap.read()
   
    if not success:
        break
   
    results = model(frame, conf=0.5)
    res = results[0]
    person_count = 0
   
    for box in res.boxes:
        cls_id = int(box.cls[0]) if box.cls is not None else None
        conf = box.conf[0] if box.conf is not None else None
   
        if cls_id is not None and conf is not None:
            class_name = res.names[cls_id]
            if class_name == 'person':
                person_count += 1
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
   
    cv2.putText(frame, f"Person Count: {person_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
   
    cv2.imshow('Frame', frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
