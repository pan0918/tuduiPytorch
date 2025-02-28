from torch.utils.data import Dataset
from PIL import Image
import os


class Mydata(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir        # 图片存放路径根目录
        self.label_dir = label_dir      # 标签存放路径根目录
        self.path = os.path.join(self.root_dir, self.label_dir)     # 整合图片路径和根路径
        self.img_path = os.listdir(self.path)       # 存放所有图片的列表

    def __getitem__(self, idx):
        img_name = self.img_path[idx]       # 获取图片名称
        img_item_dir = os.path.join(self.path, img_name)        # 合并路径
        img = Image.open(img_item_dir)      # 获取图片类
        label = self.label_dir

        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = Mydata(root_dir, ants_label_dir)     # 蚂蚁数据集
bees_dataset = Mydata(root_dir, bees_label_dir)     # 蜜蜂数据集
img, label = bees_dataset[2]
img.show()


