import os

directory_name="C:\\Users\\Pranjali\\Desktop\\Project\\Images"

directory = os.fsencode(directory_name)
a = 0
for file in os.listdir(directory):
    image_gray = os.fsdecode(file)
    if image_gray.endswith(".jpg"):
									a=(a+1)
									print(a)