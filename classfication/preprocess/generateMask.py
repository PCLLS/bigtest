import os,glob
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

from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,ALL_COMPLETED
import concurrent.futures
import logging

def generate_mask(tif,anno_path,mask_path,camelyon17_type_mask,remove=True):
    reader = mir.MultiResolutionImageReader()
    basename = osp.basename(tif)
    output_path = osp.join(mask_path, basename)
    samplename=basename.rstrip('.tif')
    xml_path=osp.join(anno_path, f'{samplename}.xml')
    if not os.path.exists(xml_path): #检查无效的Mask
        logging.info(f'check {samplename}.xml,it cannot be load!')
        if os.path.exists(output_path) and remove:
            os.system(f'rm {output_path}')
        return 1
    if osp.exists(output_path): # 跳过已有的Mask
        logging.info(f'{output_path} already exists, pass')
        return 1
    mr_image = reader.open(tif)
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(xml_path)
    xml_repository.load() 
    annotation_mask = mir.AnnotationToMask()
    label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 1, '_1': 1, '_2': 0}
    conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else ['_0', '_1', '_2']
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map,
                            conversion_order)
    logging.info(f'generate mask tif for {basename}')
    return 0

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,filename=os.path.join(MASK_FOLDER,'log.txt'))
    train_tifs = glob.glob(os.path.join(TRAINSET,'*/*.tif'))
    test_tifs = glob.glob(os.path.join(TESTSET,'*/*.tif'))
    with ProcessPoolExecutor(max_workers=20) as pool:
        futures= [pool.submit(generate_mask, tif,TRAINANNO,MASK_FOLDER,camelyon17_type_mask) for tif in sorted(train_tifs)] + [pool.submit(generate_mask, tif,TESTSETANNO,MASK_FOLDER,camelyon17_type_mask) for tif in sorted(test_tifs)]
    concurrent.futures.wait(futures, return_when=ALL_COMPLETED)
    print(f'test {len(test_tifs)}; train {len(train_tifs)}')
    for future in futures:
        print(future.result())
    