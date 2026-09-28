from .base_dataset import BaseDataset
from PIL import Image
import os
import numpy as np
class LoveDA(BaseDataset):
    def __init__(self,data_root='data/vaihingen', mode='train', transform=None,img_dir='images_png', mask_dir='masks_png', img_suffix='.png', mask_suffix='.png', domains=('Urban', 'Rural'), **kwargs):
        super(LoveDA, self).__init__(transform)

        # domains: which LoveDA scenes to load, e.g. ['Urban'] for a cross-domain (Urban -> Rural) experiment
        self.domains = list(domains)
        assert len(self.domains) > 0 and all(d in ('Urban', 'Rural') for d in self.domains), self.domains

        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.mode = mode

        if mode == "train":
            self.data_root = data_root + "/Train"
        elif mode == "val":
            self.data_root = data_root + "/Val"
        elif mode == "test":
            self.data_root = data_root + "/Test"

        self.file_paths = self.get_path(self.data_root,img_dir,mask_dir)


        #RGB
        self.color_map = {
            'building' : np.array([255, 0, 0]),  # label 0
            'road' : np.array([255, 255, 0]),  # label 1
            'water' : np.array([0, 0, 255]),  # label 2
            'barren' : np.array([159, 129, 183]),  # label 3
            'forest' : np.array([0, 255, 0]),  # label 4
            'agricultural' : np.array([255, 195, 128]),  # label 5
            'background' : np.array([255, 255, 255]),  # label 6
        }

        self.num_classes = 7

    def get_path(self, data_root, img_dir, mask_dir):
        img_ids = []
        for domain in self.domains:
            img_filename_list = os.listdir(os.path.join(data_root, domain, img_dir))
            if self.mode != 'test':
                mask_filename_list = os.listdir(os.path.join(data_root, domain, mask_dir))
                assert len(img_filename_list) == len(mask_filename_list)
            img_ids += [(str(id.split('.')[0]), domain) for id in img_filename_list]
        return img_ids


    def load_img_and_mask(self, index):
        img_id, img_type = self.file_paths[index]
        img_name = os.path.join(self.data_root, img_type, self.img_dir, img_id + self.img_suffix)
        img = Image.open(img_name).convert('RGB')

        if self.mode == 'test':
            mask = np.array(img)
            mask = Image.fromarray(mask).convert('L')
            return [img, mask, img_id]

        mask_name = os.path.join(self.data_root, img_type, self.mask_dir, img_id + self.mask_suffix)
        mask = Image.open(mask_name).convert('L')

        np_mask = np.array(mask)
        np_mask[np_mask == 0] = 8
        np_mask -= 1
        mask = Image.fromarray(np_mask).convert('L')
        return [img, mask, img_id]
