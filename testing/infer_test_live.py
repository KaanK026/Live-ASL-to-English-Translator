#do the imports
import cv2
from src.utils import load_idx_to_class
import torch
from src.Datasets import transform
from PIL import Image
from torchvision import transforms
from textblob import TextBlob
import mss
import numpy as np

idx_to_class=load_idx_to_class()

# Function to start the live inference process
def start_process(model,device):

    model.to(device)
    model.eval()

    STABILITY_THRESHOLD = 5#optimize later
    text=""
    previous_prediction=None
    same_count=0

    try:
        with mss.mss() as sct:
            monitor = {"top": 100, "left": 100, "width": 640, "height": 480}  # adjust to Zoom window
            #TODO add dynamic cropping later
            while True:
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)  # RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)


                frame_bgr=frame
                frame=transforms.ToPILImage()(frame)
                frame=transform(frame)
                input_tensor=frame.unsqueeze(0).to(device)



                with torch.no_grad():
                    output = model(input_tensor)
                    _, predicted = torch.max(output, 1)
                    predicted=predicted.item()

                predicted_class=idx_to_class.get(str(predicted), "Unknown")



                #first see if the model works letter by letter
                cv2.putText(frame_bgr,f'predicted: {predicted_class}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # we will deliver the result to the user
                if predicted_class==previous_prediction:
                    same_count+=1
                else:
                    previous_prediction=predicted_class
                    same_count=0
                if same_count>=STABILITY_THRESHOLD:
                    if predicted_class=='space':
                        #auto correction
                        if text and not text.endswith(" "):
                            last_word=text.split()[-1]
                            corrected=str(TextBlob(last_word).correct())
                            text=text[:-(len(last_word))] + corrected
                            text+=" "
                            cv2.putText(frame_bgr, f'Corrected: {corrected}', (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    elif predicted_class=='del':
                        text=text[:-1]  # Remove last character
                    elif predicted_class!='nothing':
                        text+=predicted_class



                cv2.imshow("Cam",frame_bgr)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        print("error occured.")
        cv2.destroyAllWindows()



#if __name__=="__main__":   #for testing
 #   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #  model = torch.load('model.pth', map_location=device)

   # start_process(model, device)