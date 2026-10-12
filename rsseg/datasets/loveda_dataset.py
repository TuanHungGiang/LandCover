from .base_dataset import BaseDataset
from PIL import Image
import os
import hashlib
import numpy as np
class LoveDA(BaseDataset):
    def __init__(self,data_root='data/vaihingen', mode='train', transform=None,img_dir='images_png', mask_dir='masks_png', img_suffix='.png', mask_suffix='.png', domains=('Urban', 'Rural'),
                 source_split=None, split_role='all', holdout_fraction=0.0, holdout_seed=2026,
                 num_folds=1, fold_index=0, **kwargs):
        super(LoveDA, self).__init__(transform)

        # domains: which LoveDA scenes to load, e.g. ['Urban'] for a cross-domain (Urban -> Rural) experiment
        self.domains = list(domains)
        assert len(self.domains) > 0 and all(d in ('Urban', 'Rural') for d in self.domains), self.domains

        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.mode = mode

        default_split = {'train': 'Train', 'val': 'Val', 'test': 'Test'}[mode]
        source_split = source_split or default_split
        if source_split not in ('Train', 'Val', 'Test'):
            raise ValueError(f'Unknown LoveDA source_split={source_split!r}')
        if split_role not in ('all', 'fit', 'holdout'):
            raise ValueError(f'split_role must be all|fit|holdout, got {split_role!r}')
        if split_role != 'all' and source_split != 'Train':
            raise ValueError('fit/holdout partitioning requires source_split=Train')
        if split_role != 'all' and num_folds < 2 and not 0. < holdout_fraction < 1.:
            raise ValueError('single-holdout partitioning requires holdout_fraction in (0,1)')
        if num_folds < 1 or not 0 <= fold_index < num_folds:
            raise ValueError(f'invalid fold_index={fold_index} for num_folds={num_folds}')
        self.data_root = os.path.join(data_root, source_split)
        self.source_split = source_split
        self.split_role = split_role
        self.holdout_fraction = float(holdout_fraction)
        self.holdout_seed = int(holdout_seed)
        self.num_folds = int(num_folds)
        self.fold_index = int(fold_index)

        self.file_paths = self.get_path(self.data_root,img_dir,mask_dir)


        #RGB
        self.color_map = {
            'background' : np.array([255, 255, 255]),  # label 0
            'building' : np.array([255, 0, 0]),  # label 1
            'road' : np.array([255, 255, 0]),  # label 2
            'water' : np.array([0, 0, 255]),  # label 3
            'barren' : np.array([159, 129, 183]),  # label 4
            'forest' : np.array([0, 255, 0]),  # label 5
            'agricultural' : np.array([255, 195, 128]),  # label 6
        }

        self.num_classes = 7

    @staticmethod
    def partition_ids(domain_ids, domain, split_role, holdout_fraction, holdout_seed,
                      num_folds=1, fold_index=0):
        """Domain-stratified deterministic partition used by policy search."""
        domain_ids = sorted(domain_ids)
        if split_role == 'all':
            return domain_ids
        scored = sorted(domain_ids, key=lambda image_id: hashlib.sha256(
            f'{holdout_seed}:{domain}:{image_id}'.encode()).digest())
        if num_folds >= 2:
            held_out = {image_id for rank, image_id in enumerate(scored) if rank % num_folds == fold_index}
        else:
            n_holdout = max(1, min(len(scored) - 1, int(round(holdout_fraction * len(scored)))))
            held_out = set(scored[:n_holdout])
        return [image_id for image_id in domain_ids
                if (image_id in held_out) == (split_role == 'holdout')]

    def get_path(self, data_root, img_dir, mask_dir):
        img_ids = []
        for domain in self.domains:
            img_filename_list = sorted(os.listdir(os.path.join(data_root, domain, img_dir)))
            if self.mode != 'test':
                mask_filename_list = os.listdir(os.path.join(data_root, domain, mask_dir))
                assert len(img_filename_list) == len(mask_filename_list)
            domain_ids = self.partition_ids(
                [str(name.rsplit('.', 1)[0]) for name in img_filename_list], domain,
                self.split_role, self.holdout_fraction, self.holdout_seed,
                self.num_folds, self.fold_index)
            img_ids += [(image_id, domain) for image_id in domain_ids]
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
