from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import pyramid_expand, pyramid_reduce, resize
from skimage.filters import gaussian
import sys
import imageio
import os
import cv2
from datetime import datetime 


def blend(img1, img2, mask, depth=4):
  gaussian1 = [img1]
  for i in range(depth):
    gaussian1.append(pyramid_reduce(gaussian1[i], multichannel=True, sigma=2))  

  gaussian2 = [img2]
  for i in range(depth):
    gaussian2.append(pyramid_reduce(gaussian2[i], multichannel=True, sigma=2))
  
  
  mask = gaussian(mask, multichannel=True, sigma=20)
  mask_piramid = [mask]
  for i in range(depth):
    mask_piramid.append(pyramid_reduce(mask_piramid[i], multichannel=True, sigma=10))

  reconstructed1 = [gaussian1[-1]]
  for i in range(0, len(gaussian1)-1):
    reconstructed1.append(pyramid_expand(reconstructed1[i], multichannel=True))
  reconstructed1.reverse()

  reconstructed2 = [gaussian2[-1]]
  for i in range(0, len(gaussian2)-1):
    reconstructed2.append(pyramid_expand(reconstructed2[i], multichannel=True))
  reconstructed2.reverse()

  laplacian1 = []
  for i in range(depth+1):
    laplacian1.append(gaussian1[i] - reconstructed1[i])  

  laplacian2 = []
  for i in range(depth+1):
    laplacian2.append(gaussian2[i] - reconstructed2[i])

  assert len(gaussian1) == len(gaussian2) == len(mask_piramid) == len(laplacian1) == len(laplacian2)

  blended_piramid = []
  for i in range(len(mask_piramid)-1, -1, -1):
    blended_piramid.append((laplacian1[i]*mask_piramid[i]) + ((1-mask_piramid[i])*laplacian2[i]))


  first = (mask_piramid[-1]*reconstructed1[-1]) + ((1-mask_piramid[-1])*reconstructed2[-1])
  final = blended_piramid[0] + first
  for i in range(1, depth+1):
    final = pyramid_expand(final, multichannel=True, sigma=2)
    final = final + blended_piramid[i] 


  return final


img1 = imageio.imread(sys.argv[1])
img2 = imageio.imread(sys.argv[2])



img1 = resize(img1, (512, 512))
img2 = resize(img2, (512, 512))


drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None

# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(mask,(pt1_x,pt1_y),(x,y),color=(1,1,1),thickness=20)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(mask,(pt1_x,pt1_y),(x,y),color=(1,1,1),thickness=20)        


if len(sys.argv)>3:
    mask = imageio.imread(sys.argv[3])
    mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    mask[:, :int(mask.shape[1]/2), :] = 1
else:
    print("Draw the mask and press ENTER")
    mask = np.zeros((512,512,3))
    cv2.namedWindow("Draw the mask and press ENTER")
    cv2.setMouseCallback("Draw the mask and press ENTER",line_drawing)

    while(1):
        cv2.imshow("Draw the mask and press ENTER",mask)
        if cv2.waitKey(1) & 0xFF == 13:
            break

    cv2.destroyAllWindows()

blended = blend(img1, img2, mask)
blended = blended.clip(0, 1)

plt.imshow(blended)
plt.show()

blended = (blended*255).astype('uint8')
mask = (mask*255).astype('uint8')

date = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
imageio.imwrite(os.path.join('blended_images', 'blended_'+date+".jpg"), blended)
imageio.imwrite(os.path.join('masks', 'mask_'+date+".jpg"), mask)

