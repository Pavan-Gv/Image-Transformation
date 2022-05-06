# Image-Transformation
## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
Step1:<br>
Import the necessary libraries and read the original image and save it as a image variable.
<br>
Step2:<br>
Translate the image using M=np.float32([[1,0,20],[0,1,50],[0,0,1]]) translated_img=cv2.warpPerspective(input_img,M,(cols,rows))
<br>
Step3:<br>
Scale the image using M=np.float32([[1.5,0,0],[0,2,0],[0,0,1]]) scaled_img=cv2.warpPerspective(input_img,M,(cols,rows))
<br>
Step4:<br>
Shear the image using M_x=np.float32([[1,0.2,0],[0,1,0],[0,0,1]]) sheared_img_xaxis=cv2.warpPerspective(input_img,M_x,(cols,rows))
<br>
Step5:<br>
Reflection of image can be achieved through the code M_x=np.float32([[1,0,0],[0,-1,rows],[0,0,1]]) reflected_img_xaxis=cv2.warpPerspective(input_img,M_x,(cols,rows))
<br>
Step6:<br>
Rotate the image using angle=np.radians(45) M=np.float32([[np.cos(angle),-(np.sin(angle)),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]) rotated_img=cv2.warpPerspective(input_img,M,(cols,rows))
<br>
Step7:<br>
Crop the image using cropped_img=input_img[20:150,60:230]
<br>
Step8:<br>
Display all the Transformed images and end the program.
<br>
## Program:
```python
Developed By: G Venkata Pavan Kumar.
Register Number: 212221240013
i)Image Translation
import numpy as np
import cv2
import matplotlib.pyplot as plt
inputImage=cv2.imread("gray.jpg")
inputImage=cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(inputImage)
plt.show()
rows, cols, dim = inputImage.shape
M= np.float32([[1, 0, 100],
                [0, 1, 200],
                 [0, 0, 1]])
translatedImage =cv2.warpPerspective (inputImage, M, (cols, rows))
plt.imshow(translatedImage)
plt.show()


ii) Image Scaling
import numpy as np
import cv2
import matplotlib.pyplot as plt
inputImage=cv2.imread("gray.jpg")
inputImage=cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(inputImage)
plt.show()
rows, cols, dim = inputImage.shape
M = np. float32 ([[1.5, 0 ,0],
                 [0, 1.8, 0],
                  [0, 0, 1]])
scaledImage=cv2.warpPerspective(inputImage, M, (cols * 2, rows * 2))
plt.imshow(scaledImage)
plt.show()


iii)Image shearing
import numpy as np
import cv2
import matplotlib.pyplot as plt
inputImage=cv2.imread("gray.jpg")
inputImage=cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(inputImage)
plt.show()
rows, cols, dim = inputImage.shape
matrixX = np.float32([[1, 0.5, 0],
                      [0, 1 ,0],
                      [0, 0, 1]])

matrixY = np.float32([[1, 0, 0],
                      [0.5, 1, 0],
                      [0, 0, 1]])
shearedXaxis = cv2.warpPerspective (inputImage, matrixX, (int(cols * 1.5), int (rows * 1.5)))
shearedYaxis = cv2.warpPerspective (inputImage, matrixY, (int (cols * 1.5), int (rows * 1.5)))
plt.imshow(shearedXaxis)
plt.show()
plt.imshow(shearedYaxis)
plt.show()


iv)Image Reflection

import numpy as np
import cv2
import matplotlib.pyplot as plt
inputImage=cv2.imread("gray.jpg")
inputImage=cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(inputImage)
plt.show()
rows, cols, dim = inputImage.shape
matrixx=np.float32([[1, 0, 0],
                    [0,-1,rows],
                    [0,0,1]])
matrixy=np.float32([[-1, 0, cols],
                    [0,1,0],
                    [0,0,1]])
reflectedX=cv2.warpPerspective(inputImage, matrixx, (cols, rows))
reflectedY=cv2.warpPerspective(inputImage, matrixy, (cols, rows))
plt.imshow(reflectedY)
plt.show()


v)Image Rotation

angle=np.radians(45)
inputImage=cv2.imread("gray.jpg")
M=np.float32([[np.cos(angle),-(np.sin(angle)),0],
               [np.sin(angle),np.cos(angle),0],
               [0,0,1]])
rotatedImage = cv2.warpPerspective(inputImage,M,(int(cols),int(rows)))
plt.axis('off')
plt.imshow(rotatedImage)
plt.show()


vi)Image Cropping

angle=np.radians(45)
inputImage=cv2.imread("gray.jpg")
CroppedImage= inputImage[20:150, 60:230]
plt.axis('off')
plt.imshow(CroppedImage)
plt.show()
```
## Output:
### i)Image Translation
![dip5 1](https://user-images.githubusercontent.com/94827772/167064533-ec21fa26-cf9a-4307-b47f-dce47a6166eb.png)


### ii) Image Scaling
![dip5 2](https://user-images.githubusercontent.com/94827772/167064564-6e9f86ff-8680-4eb6-8530-0651660f2ceb.png)


### iii)Image shearing
![dip5 31](https://user-images.githubusercontent.com/94827772/167064593-d64cfac0-ccac-45d0-963c-da282258d2b4.png)
![dip5 32](https://user-images.githubusercontent.com/94827772/167064595-b7332d37-76b4-444f-811a-985d5e33ff67.png)


### iv)Image Reflection
![dip5 4](https://user-images.githubusercontent.com/94827772/167064620-f3c48dec-f7d4-4e28-8cc2-16a5bf0a4608.png)


### v)Image Rotation
![dip5 5](https://user-images.githubusercontent.com/94827772/167064644-521a280e-3b84-49d3-a0c8-9ad6f1f49be5.png)


### vi)Image Cropping
![dip5 6](https://user-images.githubusercontent.com/94827772/167064655-be8d0cd6-910a-42a5-9b49-ddb0a348d8ec.png)



## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
