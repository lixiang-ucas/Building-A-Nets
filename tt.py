import cv2
import glob

imgs = glob.glob('out2/*.png')
for img in imgs:
	im = cv2.imread(img)
	fn = os.path.basename(img)
	fn = fn.split('0_')[1]
	fn = fn.split('.png')[0]
	cv2.imwrite('out_labels2/'+fn,im[:2])