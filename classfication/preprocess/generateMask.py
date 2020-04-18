import os, glob
import os.path as osp

# add classfication
BASE_DIR = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
import sys

sys.path.append(BASE_DIR)
# add ASAP into python environment
sys.path.append('/opt/ASAP/bin')
from multiprocessing import Pool
import multiresolutionimageinterface as mir
from classfication.utils.config import *
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, ALL_COMPLETED
import concurrent.futures
import logging, openslide
import numpy as np

def generate_mask(tif, anno_xml, mask_path, camelyon17_type_mask, remove=True):
    reader = mir.MultiResolutionImageReader()
    basename = osp.basename(tif)
    output_path = osp.join(mask_path, basename)
    samplename = basename.rstrip('.tif')
    xml_path = osp.join(anno_xml, f'{samplename}.xml')
    if not os.path.exists(xml_path):  # 检查无效的Mask
        if os.path.exists(output_path) and remove:
            os.system(f'rm {output_path}')
            logging.info(f'check {samplename}.xml,it cannot be load!removed')
        return 1
    elif osp.exists(output_path):  # 跳过已有的Mask
        try:
            mask_slide = openslide.OpenSlide(output_path)  # 检查完整性
            mask_slide.close()
            return 1
        except:
            logging.info(f'{output_path} broken, pass')
            os.system(f'rm {output_path}')
    mr_image = reader.open(tif)
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(xml_path)
    xml_repository.load()
    annotation_mask = mir.AnnotationToMask()
    label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 1, '_1': 1, '_2': 0, 'Exclusion': 0,
                                                                             'Tumor': 1, '_2': 0}
    conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else ['_0', '_1', '_2']
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map,
                            conversion_order)
    logging.info(f'generate mask tif for {basename}')
    return samplename

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, filename=os.path.join(MASK_FOLDER, 'log.txt'))
    train_tifs = glob.glob(os.path.join(TRAINSET, '*/*.tif'))
    test_tifs = glob.glob(os.path.join(TESTSET, '*/*.tif'))
    with ProcessPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(generate_mask, tif, TRAINANNO, MASK_FOLDER, camelyon17_type_mask) for tif in
                   sorted(train_tifs)] + [
                      pool.submit(generate_mask, tif, TESTSETANNO, MASK_FOLDER, camelyon17_type_mask) for tif in
                      sorted(test_tifs)]
    concurrent.futures.wait(futures, return_when=ALL_COMPLETED)
    print(f'test {len(test_tifs)}; train {len(train_tifs)}')
    for future in futures:
        if future != 1:
            print(future.result())