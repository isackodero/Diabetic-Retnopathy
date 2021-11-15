#way to upload image: endpoint
#way to save the image
#function to make prediction on the image
#show the results
import torch
import torchvision
from torchvision import  transforms

from PIL import Image
import io
import os
from flask import Flask, request, render_template

app= Flask(__name__)
UPLOAD_FOLDER="/home/isack/Desktop/isack/Diabetic Retnopathy/flask_API/static"
#our class map
class_map={
    0:"Grade 0",
    1:"Grade 1",
    2:"Grade 2",
    3:"Grade 3",
    4:"Grade 4",
}
MODEL_PATH="/home/isack/Desktop/isack/Diabetic Retnopathy/saved_models/model_resnet101.pth"

def transform_image(image_byte):
    my_transform=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image=Image.open(io.BytesIO(image_byte))
    return my_transform(image).unsqueeze(0)



def get_predict(model_path, image_byte,class_map):

    tensor=transform_image(image_byte=image_byte)

    my_model=torch.load(model_path)
    my_model.eval()
    
    
    outputs=my_model(tensor)
    _,pred=torch.max(outputs, 1)
    pred_idx=pred.item()

    class_name=class_map[pred_idx]
   
    return (pred_idx, class_name)



@app.route("/", methods=['GET', 'POST'])
def upload_predict():
    if request.method =="POST":
        image_file=request.files["image"]
    
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            with open(image_location, 'rb') as f:
                img_byte=f.read()
                pred_idx, class_name=get_predict(MODEL_PATH, img_byte, class_map)

            return render_template('result.html', prediction=pred_idx, image=class_name, image_loc=image_file.filename)

        
    return render_template('index.html', prediction= "No Prediction", image=None, image_loc=None)

    
if __name__=='__main__':
    app.run(port= 13000, debug=True)