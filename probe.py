from ultralytics import YOLO
import matplotlib.image as mpimg
import numpy as np

m = YOLO('weights/best.pt')
p = 'dataset/images/train/frame_00000084.png'   # a frame you KNOW has vehicles

r_path = m.predict(p, verbose=False, conf=0.25)[0]
print('PATH  detections:', len(r_path.boxes))

img = mpimg.imread(p)
if img.dtype!=np.uint8: img=(img*255).astype(np.uint8) if img.max()<=1 else img.astype(np.uint8)
if img.ndim==3 and img.shape[2]==4: img=img[:,:,:3]
r_rgb = m.predict(img, verbose=False, conf=0.25)[0]
print('ARRAY-RGB detections:', len(r_rgb.boxes))

r_bgr = m.predict(img[:,:,::-1], verbose=False, conf=0.25)[0]
print('ARRAY-BGR detections:', len(r_bgr.boxes))