import requests
import io
import streamlit as st
from PIL import Image, ImageDraw
import random

# SentiSight API details
API_TOKEN = "gnbai9bftlb6coaejflmtdbhmo"
PROJECT_ID = 61660
MODEL_NAME = "luko-modelis-1"
HEADERS = {"X-Auth-token": API_TOKEN, "Content-Type": "application/octet-stream"}  # HTTP
URL = f"https://platform.sentisight.ai/api/predict/{PROJECT_ID}/{MODEL_NAME}/"

# Object detection
def object_detection(image):
    image_read = image.read()  # Read the image bytes
    r = requests.post(url=URL, headers=HEADERS, data=image_read)  # Request for SentiSight.ai API
    return r 

# Draw boxes on the image
def draw_boxes(image, boxes):
    try: # opening the image using the PIL library
        img = Image.open(io.BytesIO(image))  
        draw = ImageDraw.Draw(img)

        # Color assignment
        class_colors = {
            'Hard hat': (0, 121, 220),       # Azure blue for Hard hats
            'Machinery': (220, 140, 0),    # Harvest orange for Machinery
            'Safety vest': (191, 220, 0)       # Pear green for Safety vests
        }
        # If label is unrecognised, choose random color
        for box in boxes:
            class_label = box['label']
            if class_label not in class_colors:
                class_colors[class_label] = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        
        # Draw rectangles with text
        for box in boxes:
            x_0 = box['x0']
            y_0 = box['y0']
            x_1 = box['x1']
            y_1 = box['y1']
            class_label = box['label']
            score = box['score']
            color = class_colors[class_label]
            draw.rectangle([x_0, y_0, x_1, y_1], outline=color, width=3)
            draw.text((x_0+10, y_0), f"{class_label}: {score:.2f}", fill=color)
        return img
    except Exception as e: # if could not open the image
        st.error(f'Error opening image: {e}') 
        return None

# Streamlit
def main():
    st.title("SentiSight.ai Construction site Object Detection")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg"])

    if uploaded_image:
        response = object_detection(uploaded_image)

        if response.status_code == 200: # if API call was successful
            response_data = response.json() # extract data

            detected_objects = []
            for obj in response_data:
                detected_objects.append({
                    'label': obj['label'],
                    'score': obj['score'],
                    'x0': obj['x0'],
                    'y0': obj['y0'],
                    'x1': obj['x1'],
                    'y1': obj['y1']
                })  

            if detected_objects:
                st.write("Objects detected")
                image_with_boxes = draw_boxes(uploaded_image.getvalue(), detected_objects)
                if image_with_boxes:
                    st.image(image_with_boxes, caption='Image with detected objects', use_column_width=True)
                else:
                    st.warning("Failed to draw bounding boxes on the image.")
            else:
                st.warning("No objects detected in the image.")
        else:
            st.error('Error performing prediction. Status code:', response.status_code)
            st.error('Error message:', response.text)

            
if __name__ == '__main__':
    main()