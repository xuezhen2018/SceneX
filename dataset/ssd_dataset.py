import random
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils import data

class ssdDataset(data.Dataset):
    def __init__(self, root, batch, list_path, max_iters=None, crop_size=(640, 320), 
                 mean=(128,128,128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.batch = batch
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.scale = scale
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters)/len(self.img_ids)))
        self.files = []
        
        for name in self.img_ids:
            img_file = osp.join(self.root + "batch_" + str(batch) + "/images/%s.png"%name)
            lbl_file = osp.join(self.root + "batch_" + str(batch) + "/labels/%s_label.png"%name)
            self.files.append({
                    "img": img_file,
                    "lbl": lbl_file,
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
        
        image = Image.open(datafiles["img"])
        label = Image.open(datafiles["lbl"])
        name = datafiles["name"]
        
        # resize
        image = image.resize(cropsize, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        
        image = np.array(image, np.float32)
        label = np.array(label, np.float32)
        
        size = image.shape
        size_l = label.shape
        #image = image[:, :, ::-1] # change to BGR
        image -= self.mean
        image /= 128.0
        image = image.transpose((2,0,1))
        
        if self.is_mirror and random.random()<0.5:
            idx = [i for i in range(size[1]-1, -1, -1)]
            idx_l = [i for i in range(size_l[1]-1, -1, -1)]
            image = np.take(image, idx, axis=2)
            label = np.take(label, idx_l, axis=1)
        
        return image.copy(), label.copy(), name

if __name__ == "__main__":
    ssd = ssdDataset("C:/unity_images/", 1, "./ssd_list/train.txt")
    for i in range(5):
        img, lbl, _ = ssd[i]
        print(img.shape, lbl.shape, np.max(img), np.min(img), np.unique(lbl))