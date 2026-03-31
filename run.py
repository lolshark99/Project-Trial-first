import torch
import cv2
from torchvision import transforms
from model import EmotionModel

mean_list_train = [0.5456, 0.4975, 0.4794]   
std_list_train  = [0.1993, 0.1924, 0.1891]   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionModel(num_classes=7)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

class_names = ['angry','disgust','fear','happy','neutral','sad','surprise']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96,96)),
    transforms.ToTensor(),
    transforms.Normalize(mean_list_train, std_list_train)
])

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face)

            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, 1)
        label = f"{class_names[predicted.item()]} ({confidence.item()*100:.1f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0), 2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()