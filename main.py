import os
import numpy as np
import torch
import cv2
from torchvision import models
from torchvision import transforms
from PIL import Image
import streamlit as st
import base64  

from ImportScript import load_models
from Face_Emotion_Classify.Model import Model

@st.cache_resource
def load_and_create_models():
    loaded_poem_generator, person_detect_model, emotion_classify_model, class_names = load_models()
    return loaded_poem_generator, person_detect_model, emotion_classify_model, class_names

loaded_poem_generator, person_detect_model, emotion_classify_model, class_names = load_and_create_models()


def classify_image(image, model=emotion_classify_model):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((96,96)),
        transforms.Normalize(mean=[0.5059, 0.4034, 0.3561], std=[0.2761, 0.2433, 0.2313]),
    ])

    input_image = transform(image).unsqueeze(0).to("cuda")

    model.eval()

    with torch.no_grad():
        output = model(input_image)

    probs = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_class_idx = probs.topk(1, dim=0)
    
    class_to_idx = {'anger': 0,'contempt': 1,'disgust': 2,'fear': 3,'happy': 4,'neutral': 5,'sad': 6, 'surprise': 7}
    idx_to_class = {idx: class_label for class_label, idx in class_to_idx.items()}
    
    label = int(top_class_idx[0].to("cpu"))
    
    return idx_to_class[label]



def generate_poem(input_text, model=loaded_poem_generator):
    mapping = {"anger":"<anger>","contempt":"<disgust>","disgust":"<disgust>","fear":"<fear>",
               "happy":"<happy>","neutral":"<neutral>","sad":"<sad>","surprise":"<surprise>"
              }
    
    replaced_text = mapping.get(input_text, input_text)
    input_prompt = "<BOS>" + " " + replaced_text
    

    poem = model(input_prompt, max_length=200, do_sample=True,
               repetition_penalty=1.1, temperature=1.2,
               top_p=0.95, top_k=50)
    
    return poem[0]["generated_text"]
    

        
def detect_and_draw_objects(frame, person_detect_model):
    results = person_detect_model(frame)
    objects = results.pred[0]
    
    for obj in objects:
        x1, y1, x2, y2, confidence, class_id = obj.tolist()
        pt1 = (int(x1 - 5), int(y1 - 25))
        pt2 = (int(x2 + 5), int(y2))
        thickness = 2
        color = (168, 76, 50)
        cv2.rectangle(frame, pt1, pt2, color, thickness)
    
    return frame, objects, x1, x2, y1, y2


def capture_camera(person_detect_model):
    capture = cv2.VideoCapture(0)
    button_counter = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        frame_with_objects, objects, x1, x2, y1, y2 = detect_and_draw_objects(frame, person_detect_model)
        cv2.namedWindow("Emotion Detection")

        yield frame_with_objects, x1, x2, y1, y2
        
        
def save_image(frame, x1, x2, y1, y2):
    detected_rect = frame[int(y1):int(y2), int(x1):int(x2)]
    cv2.imwrite("captured_rectangle.jpg", detected_rect)
    print("Detected rectangle captured!")
    cv2.destroyAllWindows()
    
    
    
def main():
    top_section_style = """
        <style>
        .top-section {
            padding: 20px;
            color: black;
            background-color: #EADBC8;
            border-radius: 50px;
        }
        .top-section h1 {
            color: #102C57;
        }
        .top-section strong {
            color: #102C57;
        }
        </style>
        """
    button_style = """
        <style>
        .stButton>button {
             background-color: #102C57;
             border-radius: 50px;
             border: none;
             color: #F8F0E5;
             padding: 20px;
             font-size: 40px;
             text-align: center}
        </style>
        """ 
    
    st.markdown(top_section_style, unsafe_allow_html=True)
    st.markdown(button_style, unsafe_allow_html=True)

    st.markdown("<div class='top-section'><h1 style='text-align: center; color: #102C57;'>AI Poem-Expression-Generator</h1><p style='text-align: center;'>Created by <strong>Ahmed Hany Hereiz and Rania Hoassam</strong><br><br></p>", unsafe_allow_html=True)

    
    st.markdown(" ")
    st.markdown(" ")
    
    camera = None
    capture_button_counter = 0

    if st.button("Open Camera"):
        camera = capture_camera(person_detect_model)

    if camera is not None:
        frame_placeholder = st.empty()

        capture_button_counter += 1
        st.write("")
        capture_button_key = f"capture_button_{capture_button_counter}"
        capture_button_pressed = st.button("Capture Image", key=capture_button_key)

        for frame_with_objects, x1, x2, y1, y2 in camera:
            frame_bytes = cv2.imencode(".png", frame_with_objects)[1].tobytes()
            frame_placeholder.write(
            f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{base64.b64encode(frame_bytes).decode()}" alt="Camera Feed"></div>',
            unsafe_allow_html=True
        )

            if capture_button_pressed:
                save_image(frame_with_objects, x1, x2, y1, y2)
                frame_placeholder.empty()
                capture_button_pressed = False
                captured_frame = Image.open("captured_rectangle.jpg")
                classification_label = classify_image(captured_frame)
                
                poem = generate_poem(classification_label)
                st.markdown(f"<div style='border: 2px solid #102C57; padding: 10px; text-align: center; font-size: 18px; border-radius: 50px;'>you look {classification_label} 🤔</div>",
        unsafe_allow_html=True)
                st.markdown(" ")
                st.markdown(
        f"<div style='border: 2px solid #102C57; padding: 10px; text-align: center; font-size: 18px; border-radius: 50px;'>{poem}</div>",
        unsafe_allow_html=True
    )
                break
    

if __name__ == "__main__":
    main()