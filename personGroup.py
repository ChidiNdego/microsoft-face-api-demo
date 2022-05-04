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

"""
Once you've set up your images, define a label at the top of your script for the PersonGroup object you'll create.
"""

# Used in the Person Group Operations and Delete Person Group examples.
# You can call list_person_groups to print a list of preexisting PersonGroups.
# SOURCE_PERSON_GROUP_ID should be all lowercase and alphanumeric. For example, 'mygroupname' (dashes are OK).
PERSON_GROUP_ID = str(uuid.uuid4()) # assign a random ID (or name it anything)

# Used for the Delete Person Group example.
TARGET_PERSON_GROUP_ID = str(uuid.uuid4()) # assign a random ID (or name it anything)


'''
Create the PersonGroup
'''
# Create empty Person Group. Person Group ID must be lower case, alphanumeric, and/or with '-', '_'.
print('Person group:', PERSON_GROUP_ID)
face_client.person_group.create(person_group_id=PERSON_GROUP_ID, name=PERSON_GROUP_ID)

# Define woman friend
woman = face_client.person_group_person.create(PERSON_GROUP_ID, "Woman")
# Define man friend
man = face_client.person_group_person.create(PERSON_GROUP_ID, "Man")
# Define child friend
child = face_client.person_group_person.create(PERSON_GROUP_ID, "Child")


"""
Assign faces to Persons

The following code sorts your images by their prefix, detects faces, and assigns the faces to each Person object.
"""

'''
Detect faces and register to correct person
'''

# im = glob.glob("personGroup_images/*")
# print(im)

# Find all jpeg images of friends in working directory
woman_images = [file for file in glob.glob('personGroup_images/*') if file.startswith("personGroup_images\\w")]
man_images = [file for file in glob.glob('personGroup_images/*') if file.startswith("personGroup_images\\m")]
child_images = [file for file in glob.glob('personGroup_images/*') if file.startswith("personGroup_images\\ch")]

# print(woman_images)

# Add to a woman person
for image in woman_images:
    w = open(image, 'r+b')
    # Check if the image is of sufficent quality for recognition.
    sufficientQuality = True
    detected_faces = face_client.face.detect_with_stream(w, detection_model='detection_03', recognition_model='recognition_04', return_face_attributes=['qualityForRecognition'])
    for face in detected_faces:
        if face.face_attributes.quality_for_recognition != QualityForRecognition.high:
            sufficientQuality = False
            break
    if not sufficientQuality: continue
    face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, woman.person_id, open(image, 'r+b'))


# Add to a man person
for image in man_images:
    m = open(image, 'r+b')
    # Check if the image is of sufficent quality for recognition.
    sufficientQuality = True
    detected_faces = face_client.face.detect_with_stream(m, detection_model='detection_03', recognition_model='recognition_04', return_face_attributes=['qualityForRecognition'])
    for face in detected_faces:
        if face.face_attributes.quality_for_recognition != QualityForRecognition.high:
            sufficientQuality = False
            break
    if not sufficientQuality: continue
    face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, man.person_id, open(image, 'r+b'))

# Add to a child person
for image in child_images:
    ch = open(image, 'r+b')
    # Check if the image is of sufficent quality for recognition.
    sufficientQuality = True
    detected_faces = face_client.face.detect_with_stream(ch, detection_model='detection_03', recognition_model='recognition_04', return_face_attributes=['qualityForRecognition'])
    for face in detected_faces:
        if face.face_attributes.quality_for_recognition != QualityForRecognition.high:
            sufficientQuality = False
            break
    if not sufficientQuality: continue
    face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, child.person_id, open(image, 'r+b'))


"""
Tip

You can also create a PersonGroup from remote images referenced by URL. 
See the PersonGroupPersonOperations methods such as add_face_from_url.
"""


"""
Train the PersonGroup

Once you've assigned faces, you must train the PersonGroup so that it can identify the visual features associated 
with each of its Person objects. The following code calls the asynchronous train method and polls the result, 
printing the status to the console.
"""

'''
Train PersonGroup
'''
print()
print('Training the person group...')
# Train the person group
face_client.person_group.train(PERSON_GROUP_ID)

while (True):
    training_status = face_client.person_group.get_training_status(PERSON_GROUP_ID)
    print("Training status: {}.".format(training_status.status))
    print()
    if (training_status.status is TrainingStatusType.succeeded):
        break
    elif (training_status.status is TrainingStatusType.failed):
        face_client.person_group.delete(person_group_id=PERSON_GROUP_ID)
        sys.exit('Training the person group has failed.')
    time.sleep(5)

"""
Tip

The Face API runs on a set of pre-built models that are static by nature 
(the model's performance will not regress or improve as the service is run). 
The results that the model produces might change if Microsoft updates the model's backend without migrating to an 
entirely new model version. To take advantage of a newer version of a model, you can retrain your PersonGroup, 
specifying the newer model as a parameter with the same enrollment images.
"""

"""
Get a test image

The following code looks in the root of your project for an image test-image-person-group.jpg and detects the faces in the image.
"""

'''
Identify a face against a defined PersonGroup
'''
# Group image for testing against
test_image_array = glob.glob('personGroup_images\\test-image-person-group.jpg')
image = open(test_image_array[0], 'r+b')

# print('Pausing for 60 seconds to avoid triggering rate limit on free account...')
# time.sleep (60)

# Detect faces
face_ids = []
# We use detection model 3 to get better performance, recognition model 4 to support quality for recognition attribute.
faces = face_client.face.detect_with_stream(image, detection_model='detection_03', recognition_model='recognition_04', 
            return_face_attributes=['qualityForRecognition'])
for face in faces:
    # Only take the face if it is of sufficient quality.
    if face.face_attributes.quality_for_recognition == QualityForRecognition.high or face.face_attributes.quality_for_recognition == QualityForRecognition.medium:
        face_ids.append(face.face_id)


"""
Output identified face IDs

The identify method takes an array of detected faces and compares them to a PersonGroup. 
If it can match a detected face to a Person, it saves the result. This code prints detailed match results to the console.
"""

# Identify faces
results = face_client.face.identify(face_ids, PERSON_GROUP_ID)

### (BadArgument) 'recognitionModel' is incompatible
print('Identifying faces in {}'.format(os.path.basename(image.name)))
if not results:
    print('No person identified in the person group for faces from {}.'.format(os.path.basename(image.name)))
for person in results:
    if len(person.candidates) > 0:
        print('Person for face ID {} is identified in {} with a confidence of {}.'.format(person.face_id, os.path.basename(image.name), 
        person.candidates[0].confidence)) # Get topmost confidence score
    else:
        print('No person identified for face ID {} in {}.'.format(person.face_id, os.path.basename(image.name)))

"""
Verify faces
The Verify operation takes a face ID and either another face ID or a Person object and determines whether they belong to the same person. 
Verification can be used to double-check the face match returned by the Identify operation.

The following code detects faces in two source images and then verifies them against a face detected from a target image.
"""

"""
Get test images

The following code blocks declare variables that will point to the source and target images for the verification operation.
"""

# Base url for the Verify and Facelist/Large Facelist operations
IMAGE_BASE_URL = 'https://csdx.blob.core.windows.net/resources/Face/Images/'

# Create a list to hold the target photos of the same person
target_image_file_names = ['Family1-Dad1.jpg', 'Family1-Dad2.jpg']
# The source photos contain this person
source_image_file_name1 = 'Family1-Dad3.jpg'
source_image_file_name2 = 'Family1-Son1.jpg'

"""
Detect faces for verification

The following code detects faces in the source and target images and saves them to variables.
"""

# Detect face(s) from source image 1, returns a list[DetectedFaces]
# We use detection model 3 to get better performance.
detected_faces1 = face_client.face.detect_with_url(IMAGE_BASE_URL + source_image_file_name1, detection_model='detection_03')
# Add the returned face's face ID
source_image1_id = detected_faces1[0].face_id
print('{} face(s) detected from image {}.'.format(len(detected_faces1), source_image_file_name1))

# Detect face(s) from source image 2, returns a list[DetectedFaces]
detected_faces2 = face_client.face.detect_with_url(IMAGE_BASE_URL + source_image_file_name2, detection_model='detection_03')
# Add the returned face's face ID
source_image2_id = detected_faces2[0].face_id
print('{} face(s) detected from image {}.'.format(len(detected_faces2), source_image_file_name2))

# List for the target face IDs (uuids)
detected_faces_ids = []
# Detect faces from target image url list, returns a list[DetectedFaces]
for image_file_name in target_image_file_names:
    # We use detection model 3 to get better performance.
    detected_faces = face_client.face.detect_with_url(IMAGE_BASE_URL + image_file_name, detection_model='detection_03')
    # Add the returned face's face ID
    detected_faces_ids.append(detected_faces[0].face_id)
    print('{} face(s) detected from image {}.'.format(len(detected_faces), image_file_name))


"""
Get verification results

The following code compares each of the source images to the target image and prints a message indicating whether 
they belong to the same person.
"""

# Verification example for faces of the same person. The higher the confidence, the more identical the faces in the images are.
# Since target faces are the same person, in this example, we can use the 1st ID in the detected_faces_ids list to compare.
verify_result_same = face_client.face.verify_face_to_face(source_image1_id, detected_faces_ids[0])
print('Faces from {} & {} are of the same person, with confidence: {}'
    .format(source_image_file_name1, target_image_file_names[0], verify_result_same.confidence)
    if verify_result_same.is_identical
    else 'Faces from {} & {} are of a different person, with confidence: {}'
        .format(source_image_file_name1, target_image_file_names[0], verify_result_same.confidence))

# Verification example for faces of different persons.
# Since target faces are same person, in this example, we can use the 1st ID in the detected_faces_ids list to compare.
verify_result_diff = face_client.face.verify_face_to_face(source_image2_id, detected_faces_ids[0])
print('Faces from {} & {} are of the same person, with confidence: {}'
    .format(source_image_file_name2, target_image_file_names[0], verify_result_diff.confidence)
    if verify_result_diff.is_identical
    else 'Faces from {} & {} are of a different person, with confidence: {}'
        .format(source_image_file_name2, target_image_file_names[0], verify_result_diff.confidence))


