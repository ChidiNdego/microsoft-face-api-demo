import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
# To install this module, run:
# python -m pip install Pillow
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, QualityForRecognition

# Create variables for ypur resource's Azure endpoint and key

"""
Important

Go to the Azure portal. 
If the Face resource you created in the Prerequisites section deployed successfully, 
click the Go to Resource button under Next Steps. 
You can find your key and endpoint in the resource's key and endpoint page, under resource management.
"""

# This key will serve all examples in this document.
KEY = ""

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = "https://tonito-face-api-demo.cognitiveservices.azure.com/"


"""
Important

Remember to remove the key from your code when you're done, and never post it publicly. 
For production, consider using a secure way of storing and accessing your credentials. 
See the Cognitive Services security article for more information.
"""

"""
Authenticate the client

Instantiate a client with your endpoint and key. 
Create a CognitiveServicesCredentials object with your key, and use it with your endpoint to create a FaceClient object.
"""

# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

test_image_array = glob.glob('chi.png')
image = open(test_image_array[0], 'r+b')
single_image_name = os.path.basename('chi.png')

# print('Pausing for 60 seconds to avoid triggering rate limit on free account...')
# time.sleep (60)


# We use detection model 3 to get better performance, recognition model 4 to support quality for recognition attribute.
detected_faces = face_client.face.detect_with_stream(image, detection_model='detection_03', recognition_model='recognition_04', return_face_attributes=['qualityForRecognition'])

# We use detection model 3 to get better performance.
# detected_faces = face_client.face.detect_with_url(url=single_face_image_url, detection_model='detection_03')
# detected_faces = face_client.face.detect_with_stream(image=single_face_image_url, detection_model='detection_03')
if not detected_faces:
    raise Exception('No face detected from image {}'.format(single_image_name))

# Convert width height to a point in a rectangle
def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height
    
    return ((left, top), (right, bottom))

def drawFaceRectangles() :
# Download the image from the url
    # response = requests.get(single_face_image_url)
    response = image
    # img = Image.open(BytesIO(response.content))
    img = Image.open(test_image_array[0])
    img = img.convert('RGB')

# For each face returned use the face rectangle and draw a red box.
    print('Drawing rectangle around face... see popup for results.')
    draw = ImageDraw.Draw(img)
    for face in detected_faces:
        draw.rectangle(getRectangle(face), outline='red')

# Display the image in the default image browser.
    img.save("output/sample2.jpg")
    # img.show()


# Uncomment this to show the face rectangles.
drawFaceRectangles()