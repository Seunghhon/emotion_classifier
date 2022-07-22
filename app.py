class maneger():
    def __init__(self):
        from PIL import Image, ImageFont, ImageDraw
        import os
        import cv2
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision.transforms as tt
        import models
        import json
        import random
        import time

        self.Image = Image
        self.ImageFont = ImageFont
        self.ImageDraw = ImageDraw
        self.os = os
        self.cv2 = cv2
        self.np = np
        self.torch = torch
        self.nn = nn
        self.F = F
        self.tt = tt
        self.models = models
        self.json = json
        self.random = random
        self.time = time

        self.lb = ""
        self.lb_status = False
        self.tm = 0
        self.lt =  ""
        self.pin = False
        self.class_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
        self.class_labels_dict = {'기쁨': 0, '당황': 1, '분노': 2, '불안': 3, '상처': 4, '슬픔': 5, '중립': 6}
        self.count = {"기쁨": 0, "당황": 0, "분노": 0, "불안": 0, "상처": 0, "슬픔": 0, "중립": 0}
        self.face_classifier = cv2.CascadeClassifier('face_classifier.xml')
        self.display_color = (246, 189, 86)

        f=open("letters.json", "r", encoding="utf-8")
        self.letter = json.load(f)
        f.close()


        pass

    def get_letter(self):
        if self.lb == "":
            return ""
        
        return self.letter[self.lb][self.random.randint(0, len(self.letter[self.lb])-1)]

    def main(self):
        model_state = self.torch.load("model.pth", map_location=self.torch.device("cpu"))
        model = self.models.getModel("emotionnet")
        model.load_state_dict(model_state['model'])

        cap = self.cv2.VideoCapture(0)

        while True:
            #print("COUNT:", self.count, "TM: ", self.tm, "LT: ", self.lt, "LB: ", self.lb, "PIN: ", self.pin, "LB_STATUS: ", self.lb_status)
            if self.lb_status == False:
                self.count = {"기쁨": 0, "당황": 0, "분노": 0, "불안": 0, "상처": 0, "슬픔": 0, "중립": 0}
                self.tm = 0
                self.lt =  ""
                self.pin = False

            elif self.pin == False:
                #check the time is after the two seconds
                if self.time.time() - self.tm > 2:
                    print("TIMEOUT 2SEC")
                    self.tm = self.time.time()
                    #check largest value in the count
                    max_value = 0
                    for key in self.count:
                        if self.count[key] > max_value:
                            max_value = self.count[key]
                            self.lb = key
                    self.lt = self.get_letter()
                    self.pin = True

            else:
                if self.time.time() - self.tm > 10:
                    print("TIMEOUT 10SEC")
                    self.tm = 0
                    self.lt = ""
                    self.pin = False
                    self.lb_status = False
                    self.count = {"기쁨": 0, "당황": 0, "분노": 0, "불안": 0, "상처": 0, "슬픔": 0, "중립": 0}


            ret, frame = cap.read()
            frame = self.cv2.flip(frame, 1)
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            faces = self.face_classifier.detectMultiScale(gray, 1.3, 5)
            if faces == ():
                self.lb_status = False
            for (x, y, w, h) in faces:
                self.cv2.rectangle(frame, (x, y), (x+w, y+h), self.display_color, 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = self.cv2.resize(roi_gray, (48, 48),
                                    interpolation=self.cv2.INTER_AREA)

                if self.np.sum([roi_gray]) != 0:
                    roi = self.tt.functional.to_pil_image(roi_gray)
                    roi = self.tt.functional.to_grayscale(roi)
                    roi = self.tt.ToTensor()(roi).unsqueeze(0)

                    # make a prediction on the ROI
                    tensor = model(roi)
                    probs = self.torch.exp(tensor).detach().numpy()
                    prob = self.np.max(probs) * 100
                    pred = self.torch.max(tensor, dim=1)[1].tolist()
                    #append the prediction to the count
                    self.count[self.class_labels[pred[0]]] += 1
                    if self.lb_status == False:
                        self.tm = self.time.time()
                    self.lb_status = True

            
            if self.lb_status == False:
                self.lb = ""
            #print(self.lb)

        cap.release()
        cv2.destroyAllWindows()

    def softmax(self, x):
        e_x = self.np.exp(x - self.np.max(x))
        return e_x / e_x.sum()





from flask import *
from flask_compress import Compress
import time
from threading import Thread
import os
import json


compress = Compress()
app = Flask(__name__)
app.secret_key = os.urandom(12)
m = maneger()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api')
def api():
    rtn = {}
    rtn.update({"emotion": m.lb})
    rtn.update({"letter": m.lt})
    # dump with utf-8
    return json.dumps(rtn, ensure_ascii=False)

if __name__ == '__main__':
    Thread(target=m.main).start()
    app.debug = True
    app.run(host="0.0.0.0", threaded=True, port=8000, use_reloader=False)
