from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('logs')
img_path = "images/pytorch.jpg"
img = Image.open(img_path)
print(img)

# ToTensor
trans_toTensor = transforms.ToTensor()
img_tensor = trans_toTensor(img)
writer.add_image("transforms", img_tensor, global_step=1)

# normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("normalize", img_norm, global_step=1)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
print(img_resize)       # 返回的是PIL类型的
img_resize_tensor = trans_toTensor(img_resize)
writer.add_image("resize", img_resize_tensor, global_step=1)

# compose - resize - 2  缩放但不改变长宽比
trans_resize_2 = transforms.Resize(256)
trans_compose = transforms.Compose([trans_resize_2, transforms.ToTensor()])
img_resize_2 = trans_compose(img)
writer.add_image("resize", img_resize_2, global_step=3)

# RandomCrop    随机裁剪
trans_random = transforms.RandomResizedCrop(64)
trans_compose_2 = transforms.Compose([trans_random, trans_toTensor])
for i in range(10):     # 裁剪10次
    img_crop = trans_compose_2(img)
    writer.add_image("randomCrop", img_crop, global_step=i)

writer.close()

