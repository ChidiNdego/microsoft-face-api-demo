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


# Detect a face in an image that contains a single face
single_face_image_url = 'https://www.biography.com/.image/t_share/MTQ1MzAyNzYzOTgxNTE0NTEz/john-f-kennedy---mini-biography.jpg'
single_image_name = os.path.basename(single_face_image_url)

# We use detection model 3 to get better performance.
detected_faces = face_client.face.detect_with_url(url=single_face_image_url, detection_model='detection_03')
if not detected_faces:
    raise Exception('No face detected from image {}'.format(single_image_name))

# Display the detected face ID in the first single-face image.
# Face IDs are used for comparison to faces (their IDs) detected in other images.
print('Detected face ID from', single_image_name, ':')
for face in detected_faces: print (face.face_id)
print()

# Save this ID for use in Find Similar
first_image_face_ID = detected_faces[0].face_id

"""
Tip: 

TODO: You can also detect faces in a local image. See the FaceOperations methods such as detect_with_stream.
"""

"""
Display and frame faces

The following code outputs the given image to the display and draws rectangles around the faces, 
using the DetectedFace.faceRectangle property.
"""

# Detect a face in an image that contains a single face
# single_face_image_url = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'
single_face_image_url = 'https://avatars.githubusercontent.com/u/46801202?s=400&u=924588b376b1df14d00a5a291eb04f1941ae439d&v=4'
single_image_name = os.path.basename(single_face_image_url)

# We use detection model 3 to get better performance.
detected_faces = face_client.face.detect_with_url(url=single_face_image_url, detection_model='detection_03')
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
    response = requests.get(single_face_image_url)
    img = Image.open(BytesIO(response.content))

# For each face returned use the face rectangle and draw a red box.
    print('Drawing rectangle around face... see popup for results.')
    draw = ImageDraw.Draw(img)
    for face in detected_faces:
        draw.rectangle(getRectangle(face), outline='red')

# Display the image in the default image browser.
    img.save("sample1.jpg")
    # img.show()


# Uncomment this to show the face rectangles.
drawFaceRectangles()