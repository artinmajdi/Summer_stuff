
import numpy as np

a = [1,4, -6 , 2]
b = np.array(a)
print(type(a))
print(type(b))

# reader = mir.MultiResolutionImageReader()
# mr_image = reader.open('camelyon17/centre_0/patient_000_node_0.tif')
# level = 2
# ds = mr_image.getLevelDownsample(level)
# image_patch = mr_image.getUCharPatch(int(568 * ds), int(732 * ds), 300, 200, level)
#
