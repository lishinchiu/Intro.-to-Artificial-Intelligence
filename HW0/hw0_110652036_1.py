#HW0-1
from PIL import Image, ImageDraw

with open('bounding_box.txt', 'r') as file:
    lines = file.readlines()

image_path = 'image.png'
image = Image.open(image_path)

draw = ImageDraw.Draw(image)

for line in lines:
    coordinates = list(map(int, line.strip().split()))
    num_boxes = len(coordinates)//4
    for i in range(num_boxes):
        x1, y1, x2, y2 = coordinates[i*4:i*4+4]
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

image.save('hw0_110652036_1.png')