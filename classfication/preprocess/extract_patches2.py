"""这里把slide read的时候的dimention换成了mask的dimention，为了防止大小不一样"""

from openslide import open_slide, __library_version__ as openslide_lib_version, __version__ as openslide_version
import numpy as np
import random, os, glob, time
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from skimage.color import rgb2hsv
import pandas as pd
#from save_npy import get_arg

'''
args = get_arg()
# set levels: 0 to 7 (greater than 4 not recommended)
lev1 = args.lev1
lev2 = args.lev2
# set patch sizes (I don't recomment changing that, although making it smaller
# will decrease run time)
patch_size = args.patch_size
patch_centre = args.patch_centre
my_pre_process = 'rescale_' + str(lev1) + '-' + str(lev2)
# choose conv base: 'Inception' or 'my-conv-base'
my_conv_base = args.my_conv_base
model_name = 'multi-input_' + my_conv_base + '_lev' + str(lev1) + str(lev2)
is_trainval_test = args.is_trainval_test
tif_folder = args.tif_folder
mask_folder = args.mask_folder
save_folder = args.save_folder
num_per_img = args.num_per_img
num_random_sample = args.num_random_sample
'''
# =======================Collect train/val and test patches=======================

# functions provided by Joshua Gordon

# See https://openslide.org/api/python/#openslide.OpenSlide.read_region
# Note: x,y coords are with respect to level 0.

def read_slide(slide, x, y, level, width, height, as_float=False):
    """ Read a region from the slide
    Return a numpy RBG array
    """
    im = slide.read_region((x, y), level, (width, height))
    im = im.convert('RGB')  # drop the alpha channel
    if as_float:
        im = np.asarray(im, dtype=np.float32)
    else:
        im = np.asarray(im)
    assert im.shape == (height, width, 3)  # 3：rgb
    return im


'''
def find_tissue_pixels(image, intensity=0.8):
    """ Return tissue pixels for an image
    """
    im_gray = rgb2gray(image)
    assert im_gray.shape == (image.shape[0], image.shape[1])
    indices = np.where(im_gray <= intensity)  # 返回满足条件的坐标，第一个数组是第一维度的坐标，第二个2
    return zip(indices[0], indices[1])
'''


def find_tissue_pixels(image):
    """ Return tissue pixels for an image
    """
    img_RGB = np.array(image)
    img_HSV = rgb2hsv(img_RGB)
    background_R = img_RGB[:, :, 0] > 203
    background_G = img_RGB[:, :, 1] > 191
    background_B = img_RGB[:, :, 2] > 201
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > 0.1113
    '''如果仅使用用threadshold，中间会有部份白色脂肪区域被隔离'''
    rgb_min = 50
    min_R = img_RGB[:, :, 0] > rgb_min
    min_G = img_RGB[:, :, 1] > rgb_min
    min_B = img_RGB[:, :, 2] > rgb_min
    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    indices = np.where(tissue_mask == 1)
    return zip(indices[0], indices[1])


def apply_mask(im, mask, color=(1, 0, 0)):
    """ Return the mask as an image
    """
    masked = np.zeros(im.shape)
    for x, y in mask: masked[x][y] = color
    return masked


def get_patches(slide, tumor_mask, lev, x0, y0, patch_size):
    """ Get patches from a slide: RBG image, tumor mask, tissue mask CENTERED at x0, y0
    imputs:
    - slide: OpenSlide object for RGB slide images
    - tumor_mask: OpenSlide object for tumor masks
    - lev: int, target zoom level for the patches, between 0 and 7
    - x0, y0: int, pixel coordinates at level 0
    - patch_size: int, usually 299
    outputs:
    - patch_image: array, RBG image
    - patch_mask: array, tumor mask
    - patch_tissue: array, tissue mask
    """

    # calc downsample factor
    downsample_factor = 2 ** lev

    # calc new x and y so that the patch is CENTER at the input x and y
    new_x = x0 - (patch_size // 2) * downsample_factor
    new_y = y0 - (patch_size // 2) * downsample_factor

    # read RGB patch
    patch_image = read_slide(slide,
                             x=new_x,
                             y=new_y,
                             level=lev,
                             width=patch_size,
                             height=patch_size)

    # read tumor mask
    patch_mask = read_slide(tumor_mask,
                            x=new_x,
                            y=new_y,
                            level=lev,
                            width=patch_size,
                            height=patch_size)

    # 1 channel is enough for the mask
    patch_mask = patch_mask[:, :, 0]

    # make tissue mask
    tissue_pixels = find_tissue_pixels(patch_image)
    patch_tissue = apply_mask(patch_image, tissue_pixels)

    return patch_image, patch_mask, patch_tissue


def check_patch_centre(patch_mask, patch_centre):
    """ Check if there is any tumor pixel in the 128x128 centre
    inputs:
    - patch_mask: array, tumor mask
    - patch_centre: int, usually 128
    outputs: Boolean
    """

    # get patch size
    patch_size = patch_mask.shape[0]

    # get the offset to check the 128x128 centre
    offset = int((patch_size - patch_centre) / 2)

    # sum the pixels in the 128x128 centre for the tumor mask
    sum_cancers = np.sum(patch_mask[offset:offset + patch_centre, offset:offset + patch_centre])

    return sum_cancers > 0


def collect_patches(tif, mask, lev1, lev2, num_per_img, patch_size, patch_centre, save_folder, num_random_sample):
    """ Extract patches with labels from the slides in the list
    inputs:
    - tif: tif
    - mask: mask
    - lev1: int, target zoom level for the patches, between 0 and 7 - higher resolution: lev1<lev2
    - lev2: int, target zoom level for the patches, between 0 and 7 - lower resolution
    - num_per_imgm: int, number of patches to extract per slide per class, usually 100
    - patch_size: int, usually 299
    - patch_centre: int, usually 128
    save_folder: save csv file path.
    outputs:
    - patch_images: list, extracted patches as arrays
    - patch_labels: list, labels of patches: 0 - healthy, 1 - tumor
    - save a plot of the patches
    """
    table=pd.DataFrame(columns=['slide_name','x','y','label'])
    # init output lists
    patch_images_lev1 = []
    patch_images_lev2 = []
    patch_labels = []

    num_cancer = 0
    num_health = 0

    # file paths
    slide_path = tif
    mask_path = mask
    f_num = slide_path.split('/')[-1].split('.')[0]
    slide_name=os.path.basename(slide_path).rstrip('.tif')

    # get images with OpenSlide
    slide = open_slide(slide_path)
    tumor_mask = open_slide(mask_path)

    # read level 4 slide image and mask - for the purposes of getting healthy
    # and tumor pixels
    # 读取slide和mask，read_slide就是返回一shape == (height, width, 3) #3：rgb
    slide_image = read_slide(slide,
                             x=0,
                             y=0,
                             level=4,
                             width=tumor_mask.level_dimensions[4][0],
                             height=tumor_mask.level_dimensions[4][1])

    mask_image = read_slide(tumor_mask,
                            x=0,
                            y=0,
                            level=4,
                            width=tumor_mask.level_dimensions[4][0],
                            height=tumor_mask.level_dimensions[4][1])



    print('--------checking mask image shape after read slide', mask_image.shape)
    print('--------checking slide_image shape after read slide', slide_image.shape)
    mask_image = mask_image[:, :, 0]
    # print ('--------checking mask image shape after mask_image[:, :, 0]', mask_image.siz)

    # get a list of tumor pixels at level 4
    mask_lev_4_cancer = np.nonzero(mask_image)
    # print ('checking length of mask_lev_4_cancer', mask_lev_4_cancer)

    # make a healthy tissue mask by subtracting tumor mask from tissue mask
    tissue_pixels = find_tissue_pixels(slide_image)
    # print ('---checking tissue_pixels ', tissue_pixels )
    tissue_regions = apply_mask(slide_image, tissue_pixels)
    # print ('------checking tissue_regions', tissue_regions)

    mask_health = tissue_regions[:, :, 0] - mask_image
    # print ('------checking mask_health = tissue_regions[:, :, 0] - mask_image-------', mask_health.shape)
    mask_health = mask_health > 0
    # print ('------checking mask_health = mask_health > 0---------', mask_health.shape)
    mask_health = mask_health.astype('int')
    # print ('------checking mask_health = mask_health.astypeint-------', mask_health.shape)

    # get a list of healthy pixels at level 4
    mask_lev_4_health = np.nonzero(mask_health)
    # print ('------checking mask_lev_4_health----', len(mask_lev_4_health[0]))

    # print()
    # print('lenmask_lev_4_cancerpatch_size ** 2, lenmask_lev_4_health0patch_size ** 2:',
    # len(mask_lev_4_cancer[0]) // (patch_size ** 2), len(mask_lev_4_health[0]) // (patch_size ** 2))

    # -------------------------------------------------------------
    if len(mask_lev_4_cancer[0]) != 0:
        print('extracting tumor patches------')
        #logging.info('extracting tumor patches')
        # extract TUMOR patches

        # get a random sample of tumor pixels
        # Note: did random.sample here rather than random.choice inside the while loop because os speed
        random_sample = min(len(list(zip(mask_lev_4_cancer[1], mask_lev_4_cancer[0])))-1,num_random_sample)
        sample_cancer = random.sample(list(zip(mask_lev_4_cancer[1], mask_lev_4_cancer[0])), random_sample)

        c = 0
        idx= 0
        # continue until enough patches extracted
        while num_cancer < num_per_img:
            c += 1
            if c == random_sample:
                break
            # print('-----checking-------c', c)
            # if c % 10 == 0:
            # print(c, end=', ')

            # get the next pixel from the sample - coordinates at level4
            (x4, y4) = sample_cancer[c]

            # convert level 4 coordinates to level 0
            x0 = x4 * (2 ** 4)
            y0 = y4 * (2 ** 4)
            
            # extract patches at lev1 CENTERED at that pixel
            patch_image_lev1, patch_mask_lev1, patch_tissue_lev1 = \
                get_patches(slide, tumor_mask, lev1, x0, y0, patch_size)

            # calc tissue ratio in that patch
            tissue_ratio = np.sum(patch_tissue_lev1[:, :, 0]) / (patch_size ** 2)

            # double-check if the patch has tumor
            has_cancer = check_patch_centre(patch_mask_lev1, patch_centre)

            # if it has more than 50% tissue and has tumor
            if (tissue_ratio > 0.5) & has_cancer:
                # collect lev1 patch
                num_cancer += 1
                table.loc[idx]=(slide_name,x0,y0,1)
                idx+=1

    # -------------------------------------------------------------
    # extract HEALTHY patches
    # repeat the above for the healthy pixels
    print('extracting normal patches------')
    #logging.info('extracting normal patches')

    # print()
    # get a random sample of healthy pixels
    random_sample = min(len(list(zip(mask_lev_4_health[1], mask_lev_4_health[0])))-1, num_random_sample)
    sample_health = random.sample(list(zip(mask_lev_4_health[1], mask_lev_4_health[0])), random_sample)
    # print('-------checking sample_health------', len(sample_health))

    c = 0

    # get healthy images
    while num_health < num_per_img:
        c += 1
        if c == random_sample:
            break
        # if c % 10 == 0:
        # print(c, end=', ')

        # get the next pixel from the sample - coordinates at level 4
        (x4, y4) = sample_health[c]

        # convert level 4 coordinates to level 0
        x0 = x4 * (2 ** 4)
        y0 = y4 * (2 ** 4)

        # extract patches at lev1 CENTERED at that pixel
        patch_image_lev1, patch_mask_lev1, patch_tissue_lev1 = \
            get_patches(slide, tumor_mask, lev1, x0, y0, patch_size)

        # calc tissue ratio in that patch
        tissue_ratio = np.sum(patch_tissue_lev1[:, :, 0]) / (patch_size ** 2)

        # check if the patch has tumor
        has_cancer = check_patch_centre(patch_mask_lev1, patch_centre)

        # if it has more than 50% tissue and doens't have tumor in the 128x128 centre
        if (tissue_ratio > 0.5) & (not has_cancer):

            # collect lev1 patch
            num_health += 1
            table.loc[idx]=(slide_name,x0,y0,0)
            idx+=1
        table.to_csv(save_folder,header=True)
    return table
