import random
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils import data

class cityscapesDataset(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(640, 320),
                 mean=(128,128,128), scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.scale = scale
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters)/len(self.img_ids)))
        
        self.files = []
        self.set = set
        
        self.id_to_trainid = {7:0, 8:1, 11:2, 12:3, 13:4, 17:5,
                              19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12,
                              26:13, 27:14, 28:15, 31:16, 32:17, 33:18}
        
        for name in self.img_ids:
            img_file = osp.join(self.root, "Cityscapes/leftImg8bit/%s/%s"%(self.set, name))
            label_name = name.split('.')[0][:-11] + "gtFine_labelIds.png"
            label_file = osp.join(self.root, "Cityscapes/gtFine/%s/%s"%(self.set, label_name))
            self.files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })
    
    def __len__(self):
        return len(self.files)
    
    def __scale__(self):
        cropsize = self.crop_size
        if self.scale:
            r = random.random()
            if r>0.7:
                cropsize = (int(self.crop_size[0]*1.1), int(self.crop_size[1]*1.1))
            elif r<0.3:
                cropsize = (int(self.crop_size[0]*0.8), int(self.crop_size[1]*0.8))
        
        return cropsize
    
    def __getitem__(self, index):
        datafiles = self.files[index]
        cropsize = self.__scale__()
        
        image = Image.open(datafiles["img"]).convert("RGB")
        label = Image.open(datafiles["label"])
        name = datafiles["name"]
        
        # resize
        image = image.resize(cropsize, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        
        image = np.array(image, np.float32)
        label = np.array(label, np.float32)
        
        label_copy = 255*np.ones(label.shape, dtype=np.int32)
        for k,v in self.id_to_trainid.items():
            label_copy[label==k]=v
        label_copy = np.asarray(label_copy, np.float32)
        
        size = image.shape
        size_l = label.shape
        
        image -= self.mean
        image /= 128.0
        image = image.transpose((2,0,1))
        
        if self.is_mirror and random.random() < 0.5:
            idx = [i for i in range(size[1]-1, -1, -1)]
            idx_l = [i for i in range(size_l[1]-1, -1, -1)]
            image = np.take(image, idx, axis=2)
            label_copy = np.take(label_copy, idx_l, axis=1)
        
        return image.copy(), label_copy.copy(), name
        
if __name__ == "__main__":
    city = cityscapesDataset(root="../../dataset", list_path="./cityscapes_list/val.txt", crop_size=(1024, 512), set="val")
    
    for i in range(10):
        img, lbl, name = city[i]
        print(np.max(img), np.min(img), np.unique(lbl), name)
        
        img = img.transpose((1,2,0))
        img *= 128.0
        img += (128, 128, 128)
        
        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint8)
        
        img = Image.fromarray(img)
        lbl = Image.fromarray(lbl)
        
        img.save("%d.png"%i)
        lbl.save("%d_label.png"%i)