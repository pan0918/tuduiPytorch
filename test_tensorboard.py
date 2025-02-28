from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter('log')

img_path = 'data/train/ants_image/0013035.jpg'
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
# print(img_array.shape)
# print(type(img_array))
writer.add_image("test", img_array, 1, dataformats='HWC')

for i in range(100):
    writer.add_scalar('y = 3x', 3*i, i)

writer.close()