# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmseg.core import encode_mask_results
from collections import defaultdict
from mmseg.datasets.panopticapi.utils import get_traceback, IdGenerator, id2rgb, rgb2id, save_json
from mmseg.datasets.coco_panoptic import id_and_category_maps as coco_categories_dict
import os
import PIL.Image as Image
import json


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test_pan(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    #results = []
    results = defaultdict(lambda:[])
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        if (i==3):
            break
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        assert isinstance(result,dict)

        batch_size = len(list(result.values())[0])
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if 'bbox' in result.keys():
            results['bbox'].extend([bbox for bbox in result['bbox']])
        if 'segm' in result.keys():
            results['segm'].extend([encode_mask_results(segm) for segm in result['segm']])
        if 'panoptic' in result.keys():
            results['panoptic'].extend([panoptic for panoptic in result['panoptic']]) 

        for _ in range(batch_size):
            prog_bar.update()
    return results

OFFSET = 1000
VOID=0
def multi_gpu_test_pan(model,
                   data_loader,
                   datasets='coco',
                   segmentations_folder=None, 
                   tmpdir=None, 
                   gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    if datasets == 'coco':
        categories_dict = coco_categories_dict
    else:
        assert False
    categories = {el['id']: el for el in categories_dict}
    
    if segmentations_folder==None:
        assert False, 'segmentations_folder should not be none'
    results = {'bbox':[],'segm':[],'panoptic':[]}
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
      
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results

            assert isinstance(result,dict)
        if 'bbox' in result.keys():
            results['bbox'].extend([bbox for bbox in result['bbox']])
        if 'segm' in result.keys():
            results['segm'].extend([encode_mask_results(segm) for segm in result['segm']])
        if 'panoptic' in result.keys():
            annotations = []
            for panoptic_result in result['panoptic']:
                original_format, file_name,shape = panoptic_result
                id_and_category_maps = OFFSET * original_format[:, :, 0] + original_format[:, :, 1]
                pan_format = np.zeros((original_format.shape[0], original_format.shape[1], 3), dtype=np.uint8)
                #results['panoptic'].extend([panoptic for panoptic in result['panoptic']]) 
                id_generator = IdGenerator(categories)

                #if datasets == 'ade20k':
                #    pan_format[:,:,0] = 151
                l = np.unique(id_and_category_maps)
                segm_info = []
                for el in l:
                    sem = el // OFFSET
                    
                    if sem == VOID:
                        continue
                    if sem not in categories:
                        raise KeyError('Unknown semantic label {}'.format(sem))
                    mask = id_and_category_maps == el
                    
                    segment_id, color = id_generator.get_id_and_color(sem)
                   
                    pan_format[mask] = color

                
                    segm_info.append({"id": segment_id,
                                    "isthing": categories[int(sem)]['isthing']==1,
                                  "category_id": int(sem)
                                })
                suffix = '.png'
                if datasets in ['cityscapes', 'ade20k']:
                    #if '_' in file_name:
                    image_id = file_name.split('_')[:3]   # for cityscapes
                    image_id = '_'.join(image_id)
                elif datasets in ['mapillary']:
                    image_id = file_name.split('.')[0]
                elif datasets in ['coco']:
                    image_id = int(file_name) # for coco
                annotations.append({'image_id': image_id,
                                'file_name':file_name+suffix,
                                "segments_info": segm_info
                                })
                pan_format = pan_format[:,:,::-1]   ## note this 
                
                mmcv.imwrite(pan_format,os.path.join(segmentations_folder, file_name+suffix))
                '''
                detectron2_show = False
                if detectron2_show:
                    #print(data['img_metas'][0].data[0][0])
                    #print(segm_info)
                    try:
                        im = Image.open(data['img_metas'][0].data[0][0]['filename'])
                        meta = MetadataCatalog.get("coco_2017_val_panoptic_separated") 
                        im = np.array(im)[:, :, ::-1]
                        v = Visualizer(im, meta, scale=1.0)
                        v._default_font_size = 10
                        v = v.draw_panoptic_seg_predictions(torch.from_numpy(rgb2id(pan_format[:,:,::-1])), segm_info, area_threshold=0)
                        mmcv.imwrite(v.get_image(),os.path.join(segmentations_folder+'2', file_name+'_c'+suffix))
                    except:
                        pass
                '''
                #print(pan_format.shape,os.path.join(segmentations_folder, file_name+suffix) )
                #img = Image.fromarray(pan_format)

                #img.save(os.path.join('seg', file_name+suffix))
                
            results['panoptic'].extend([annotation for annotation in annotations])

        if rank == 0:
            batch_size = len(result[list(result.keys())[0]])
            
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    
    gpu_collect = False
    if tmpdir is None: 
        tmpdir = './tmp'
    if gpu_collect:
        results['bbox'] = collect_results_gpu(results['bbox'], len(dataset))
        results['segm'] = collect_results_gpu(results['segm'], len(dataset))
        results['panoptic'] = collect_results_gpu(results['panoptic'], len(dataset))
    else:
        if 'bbox' in results.keys():
            results['bbox'] = collect_results_cpu(results['bbox'], len(dataset), tmpdir+'_bbox')
        if 'segm' in results.keys():
            results['segm'] = collect_results_cpu(results['segm'], len(dataset), tmpdir+'_segm')
        if 'panoptic' in results.keys():
            results['panoptic'] = collect_results_cpu_panoptic(results['panoptic'], tmpdir+'_panoptic')
    #print(results,gpu_collect)

    if rank == 0:
        with open(segmentations_folder+'_annotations.json','w') as f:
            json.dump(results['panoptic'],f)

    time.sleep(1) 
    return results

def single_gpu_test_seg(model,
                        data_loader,
                        show=False,
                        out_dir=None,
                        efficient_test=False,
                        opacity=0.5):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    if efficient_test:
        mmcv.mkdir_or_exist('.efficient_test')
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file,
                    opacity=opacity)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir='.efficient_test')
            results.append(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test_seg(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    if efficient_test:
        mmcv.mkdir_or_exist('.efficient_test')
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result, tmpdir='.efficient_test')
            results.append(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu_panoptic(result_part, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        filename_set = {}
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results

