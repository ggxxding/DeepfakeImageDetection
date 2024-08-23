import os
import face_recognition

dir = '/mnt/share_data/dmj/phase2/phase2/1_fake/'
files = os.listdir(dir)

c = 0
emptylist = []
for file in files:
    image = face_recognition.load_image_file(os.path.join(dir,file))
    face_locations = face_recognition.face_locations(image)
    c+= 1
    print(c)
    if face_locations == []:
        emptylist.append(file)

with open('/home/ubuntu/dmj/DeepfakeImageDetection/data/fake.csv', 'w') as f:
    for line in emptylist:
        f.write(line + ',' + '1.0' + '\n')


# # Load the jpg file into a numpy array
# image = face_recognition.load_image_file("biden.jpg")

# # Find all the faces in the image using the default HOG-based model.
# # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# # See also: find_faces_in_picture_cnn.py
# face_locations = face_recognition.face_locations(image)

# print("I found {} face(s) in this photograph.".format(len(face_locations)))
