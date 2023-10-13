

import SimpleITK as sitk
import matplotlib.colorbar
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import scipy
import nibabel as nb
from nilearn.image import threshold_img



# Functions to use in preprocess


def img_to_array(filedir):
    '''Read image in filedir into array'''
    img = sitk.ReadImage(filedir)
    array = sitk.GetArrayFromImage(img)
    return array

def z_score_normalize(array):
    array = np.float64(array)
    mean = np.mean(array)  # mean for data_prep centering
    std = np.std(array)  # std for data_prep normalization
    array -= mean
    array /= std

    #m = np.max(array)

    return array

def max_normalize(array):
    #array = np.float64(array)
    m = np.max(array)
    array_norm = array / m
    return array_norm

def min_max_normalize(array):
    #array = np.float64(array)
    ma = np.max(array)
    mi = np.min(array)

    array_norm = (array - mi) / (ma - mi)
    return array_norm

def suvr_normalize(array, ref_mask_file):
    ref_img_mask = img_to_array(ref_mask_file)

    ref_region = array*ref_img_mask
    num_non_zero_voxels = np.count_nonzero(ref_region)

    ref_mean_value = np.sum(ref_region)/num_non_zero_voxels

    out = array/ref_mean_value

    return out



def padding(array, value):
    array = np.float64(array)
    array = np.pad(array, ((11,11), (7,7), (6,6)), constant_values=value)
    return array

def crop(array):
    array = array[2:-3,1:-2,2:-3]
    return array

def mask(array, binary_mask_file):
    mask_array = img_to_array(binary_mask_file)
    #mask_array = np.where(mask_array==label, 1, 0)
    array = array*mask_array
    return array

def get_binary_mask(labels_mask, label_needed, save_name=None):
    mask_array = img_to_array(labels_mask)
    mask_array = np.where(mask_array == label_needed, 1, 0)
    mask_array = mask_array.astype('uint8')

    if save_name:
        result_image = sitk.GetImageFromArray(mask_array)
        result_image.CopyInformation(sitk.ReadImage(labels_mask))

        sitk.WriteImage(result_image, save_name)
    return mask_array


def resize_array(array, output_dims):
    resized_array = resize(array, output_dims)
    resized_array = resized_array.astype('float32')
    return resized_array


# Visualizing data_prep
def visualize_scan(dataset, print_img=False):
    """Get dimensions of scan and labels and eventually print slices of first image of dataset"""
    data = dataset.take(1)
    images, labels = list(data)[0]
    images = images.numpy()
    image = images[0]
    print("Dimension of the scans is:", images.shape)
    print('Dimension of the labels is: ' + str(labels.shape))
    if print_img:
        create_slice_figure(image, 'random image')

    return image

def create_slice_figure(img_array, name, save_name=None):
    """ Print several slices of 3D array"""
    rotation = 90
    fig = plt.figure(figsize=(6,3.6), dpi=500)
    plt.title(name)
    plt.axis('off')
    print(img_array.shape)
    shape = int(img_array.shape[0] / 6)

    for i in range(1,6):
        fig.add_subplot(3, 6, i)
        #plt.tight_layout()
        img_slice = img_array[:,:,i*shape]
        img_slice = scipy.ndimage.rotate(img_slice, rotation)
        #matplotlib.colorbar.Colorbar(ax=0)
        plt.imshow(img_slice, cmap='jet')

        plt.axis('off')
        #plt.subplots_adjust(hspace=0.0001)

    #for i in range(6):
        fig.add_subplot(3, 6, i+6)
        #plt.tight_layout()
        img_slice = img_array[:,i*shape,:]
        img_slice = scipy.ndimage.rotate(img_slice, rotation)
        #matplotlib.colorbar.Colorbar(ax=0)
        plt.imshow(img_slice, cmap='jet')
        plt.axis('off')
        #plt.subplots_adjust(hspace=0.0001)

    #for i in range(6):
        fig.add_subplot(3, 6, i+12)
        #plt.tight_layout()
        img_slice = img_array[i*shape,:,:]
        img_slice= np.flip(img_slice, axis=0)
        img_slice = scipy.ndimage.rotate(img_slice, rotation)

        #matplotlib.colorbar.Colorbar(ax=0)
        plt.imshow(img_slice, cmap='jet')
        #plt.subplots_adjust(hspace=0.0001)

        plt.axis('off')

    plt.colorbar(cax=plt.axes([0.79, 0.16, 0.02, 0.7]), anchor=(np.min(img_array), np.max(img_array)))

    plt.subplots_adjust(wspace=0.03, hspace=0.0001)
    if save_name:
        plt.savefig(save_name)
    else:
        fig.show()
    #fig.show()

def create_slice_figure_2(img_array, name, save_name=None):
    """ Print several slices of 3D array"""
    rotation = 180
    plt.figure()
    plt.title(name)
    plt.axis('off')
    print(img_array.shape)
    shape = int(img_array.shape[0] / 6)

    for i in range(6):
        plt.subplot(3, 6, i + 1)
        # plt.tight_layout()
        img_slice = img_array[:, :, i * shape]
        img_slice = scipy.ndimage.rotate(img_slice, rotation)
        # matplotlib.colorbar.Colorbar(ax=0)
        plt.imshow(img_slice, cmap='jet')

        plt.axis('off')
        # plt.subplots_adjust(hspace=0.0001)

        # for i in range(6):
        plt.subplot(3, 6, i + 7)
        # plt.tight_layout()
        img_slice = img_array[:, i * shape, :]
        img_slice = scipy.ndimage.rotate(img_slice, rotation)
        # matplotlib.colorbar.Colorbar(ax=0)
        plt.imshow(img_slice, cmap='jet')
        plt.axis('off')
        # plt.subplots_adjust(hspace=0.0001)

        # for i in range(6):
        plt.subplot(3, 6, i + 13)
        # plt.tight_layout()
        img_slice = img_array[i * shape, :, :]
        img_slice = np.flip(img_slice, axis=0)
        # img_slice = scipy.ndimage.rotate(img_slice, rotation)

        # matplotlib.colorbar.Colorbar(ax=0)
        plt.imshow(img_slice, cmap='jet')
        # plt.subplots_adjust(hspace=0.0001)

        plt.axis('off')

    # plt.colorbar(anchor=(np.min(img_array), np.max(img_array)))

    plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.03, hspace=0.1)
    if save_name:
        plt.savefig(save_name)
    else:
        plt.show()
