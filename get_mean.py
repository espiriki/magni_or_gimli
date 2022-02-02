from PIL import Image
import numpy as np

files=["red.jpg","green.jpg","blue.jpg"]


for file in files:

    img = Image.open(file)

    img_as_np_array = np.asarray(img)

    print("mean for file {}".format(file))
    print("shape for RED {}".format(img_as_np_array[:,:,0].shape))
    print("mean R : {}".format(np.mean(img_as_np_array[:,:,0])))
    print("mean G: {}".format(np.mean(img_as_np_array[:,:,1])))
    print("mean B: {}".format(np.mean(img_as_np_array[:,:,2])))