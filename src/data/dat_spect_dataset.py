

import numpy as np
import nibabel as nb

from monai.transforms import Compose, Lambda, NormalizeIntensity, ToTensor, GaussianSmooth
from torch.utils.data import DataLoader, Dataset

#from src.utils.process_image import img_to_array, process_image, create_slice_figure


class SPECTDataset(Dataset):
    def __init__(self, data, label_dict):
        self.data = data
        self.label_dict = label_dict
        #self.max_dataset, _ = self.get_max_min_dataset()
        self.transform_gen = Compose([
            #CenterSpatialCrop(roi_size=output_shape),
            Lambda(self.crop),
            GaussianSmooth(sigma=0.7),
            Lambda(self.mni_mask),
            NormalizeIntensity(),
            ToTensor()
            ])
        self.transform_spect = Compose([
            Lambda(self.mni_mask),
            NormalizeIntensity(),
            ToTensor()
            ])

    def __getitem__(self, idx):
        # Load the scan data
        path = self.data.iloc[idx]['img_paths']
        x = nb.load(path).get_fdata()

        gen_orig = self.data.iloc[idx]['gen_orig']

        if gen_orig == 'gen':
            x_process = self.transform_gen(x)
        elif gen_orig == 'origSPECT':
            x_process = self.transform_spect(x)

        # Get the corresponding label
        y = self.data.iloc[idx]['labels']
        y_id = self.label_dict.get(y)

        return x_process, y_id

    def __len__(self):
        return len(self.data)

    def get_max_min_dataset(self):

        filelist = self.data['img_paths'].values
        arrays = [img_to_array(path) for path in filelist]

        """Get the max and min value of arrays of images in filelist"""
        max_dataset = np.max(arrays)
        min_dataset = np.min(arrays)

        print('Max value: ' + str(max_dataset))
        print('Min value: ' + str(min_dataset))

        return max_dataset, min_dataset

    @staticmethod
    def mni_mask(array, mask_file='/home/leonor/Code/brain_masks/brainmask.nii'):
        """ Apply mask to image"""
        mask_img = nb.load(mask_file)
        mask = np.array(mask_img.dataobj)

        array_masked = array * mask
        return array_masked

    @staticmethod
    def crop(array):
        array = array[2:-3, 1:-2, 2:-3]
        return array