from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image

img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

# 1.transforms的使用
tensor_trans = transforms.ToTensor()    # 相当于从transforms中取出一个工具出来
tensor_img = tensor_trans(img)          # PIL类型和numpy提供的narrays类型都可

writer = SummaryWriter("log")
writer.add_image("transform_test", tensor_img, 1)

writer.close()

# print(tensor_img.size())
# print(tensor_img)

# 2.为什么我们需要tensor数据类型


