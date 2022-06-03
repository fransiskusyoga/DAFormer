# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Modification of config and checkpoint to support legacy models

import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp
import tempfile

import mmcv
import numpy as np
import torch
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmcv.parallel import is_module_wrapper

from mmseg.utils.utils import downscale_label_ratio
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


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


def single_gpu_test(model,
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
            if is_module_wrapper(model):
                model_m = model.module
                a = model_m.extract_feat(data['img'][0].to(next(model_m.parameters()).device))
            else:
                a = model.extract_feat(data['img'][0].to(next(model_m.parameters()).device))
            result = a[-1]#.cpu().numpy()

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
    print()
    return results


def update_legacy_cfg(cfg):
    # The saved json config does not differentiate between list and tuple
    cfg.data.test.pipeline[1]['img_scale'] = tuple(
        cfg.data.test.pipeline[1]['img_scale'])
    # Support legacy checkpoints
    if cfg.model.decode_head.type == 'UniHead':
        cfg.model.decode_head.type = 'DAFormerHead'
        cfg.model.decode_head.decoder_params.fusion_cfg.pop('fusion', None)
    cfg.model.backbone.pop('ema_drop_path_rate', None)
    return cfg


def calculate_tsne(model, outputs, gt_seg_maps, min_limit=100):
    all_feat = get_feat_per_class(model, outputs, gt_seg_maps)
    n_class =  len(all_feat)

    min_feat = min([len(x) for x in all_feat])
    min_feat = max(min_limit, min_feat)
    
    all_label = []
    for i,x in enumerate(all_feat):
        if x.shape[0] > min_feat:
            all_feat[i] = x[ [j*x.shape[0]//min_feat for j in range(min_feat)] ]
        all_label.append(torch.ones([all_feat[i].shape[0]])*i)
    all_feat = torch.cat(all_feat, dim=0).cpu()
    all_label = torch.cat(all_label, dim=0).cpu()
    reduction = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        n_iter=500,
        n_iter_without_progress=150,
        n_jobs=2,
        random_state=0,
    )
    result = reduction.fit_transform(all_feat, all_label)

    _, ax = plt.subplots()
    plt_x = MinMaxScaler().fit_transform(result)
    if is_module_wrapper(model):
        model_m = model.module
        palette = model_m.PALETTE
    else:
        palette = model.PALETTE

    for digit in range(n_class):
        mask = all_label == digit
        if torch.any(mask)==False:
            continue
        clr = [x/255.0 for x in palette[digit]]
        clr.append(1.0)
        ax.scatter(
            *plt_x[mask].T,
            marker=".",#f"${digit}$",
            s=10,
            color=clr,
            alpha=0.425,
            zorder=2,
        )

    ax.set_title("Tsne plot")
    ax.axis("off")
    plt.savefig('foo.png')


def get_feat_per_class(model, outputs, gt_seg_maps):
    device = outputs.device
    prog_bar = mmcv.ProgressBar(len(outputs))

    if is_module_wrapper(model):
        model_m = model.module
        num_classes = model_m.num_classes
    else:
        num_classes = model.num_classes
    scale_factor = gt_seg_maps.shape[-1] // outputs.shape[-1]
    scale_min_ratio = 0.5
    
    all_class_feats = [ [] for i in range(num_classes)]
    for i in range(len(outputs)):
        gt = gt_seg_maps[i].clone()[None,None,:,:].to(torch.int64)
        
        gt = downscale_label_ratio(gt, 
                                   scale_factor,
                                   scale_min_ratio,
                                   num_classes,
                                   255).squeeze(0) #[1,H,W]
        ot = outputs[i].clone() #[CHN,H,W]
        all_classes = torch.unique(gt)

        # get the list of appeared class
        if (all_classes[-1] == 255): #ignore label 225
            all_classes = all_classes[:-1]
        AC = len(all_classes)
        CHN = ot.shape[0] 
            
        # generate one hot mask
        gt = gt == all_classes.reshape([-1,1,1]) #[AC,H,W]
        gt = gt.unsqueeze(1).tile([1,CHN,1,1]) #[AC,CHN,H,W]

        for i,cls in enumerate(all_classes):
            all_class_feats[cls].append(
                ot[gt[i]].reshape([CHN,-1]).transpose(0,1))
        
        prog_bar.update()
    
    for i in range(num_classes):
        if len(all_class_feats[i])!=0:
            all_class_feats[i] = torch.cat(all_class_feats[i], dim=0)
        else:
            all_class_feats[i] = torch.zeros([0,CHN]).to(device)
    print()

    return all_class_feats


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg = update_legacy_cfg(cfg)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    efficient_test = False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                efficient_test, args.opacity)
    outputs = torch.cat(outputs, dim=0)

    gt_seg_maps = dataset.get_gt_seg_maps(efficient_test)
    gt_seg_maps = np.concatenate([np.expand_dims(x, 0) for x in gt_seg_maps])
    gt_seg_maps =  torch.from_numpy(gt_seg_maps).to(outputs.device)
    
    calculate_tsne(model, outputs, gt_seg_maps)


if __name__ == '__main__':
    main()
