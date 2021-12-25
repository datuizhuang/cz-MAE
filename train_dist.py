import argparse
import os
import glob
import time
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from MAE.logger import setup_logger
from MAE.utils import AverageMeter
from MAE.lr_scheduler import get_scheduler
from MAE.transform import getTrans
from MAE.ImageNetDataset import MyDataSet, DataPrefetcher
from MAE import models
from MAE.utils import AdamW
from MAE.utils import gpu_listener

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from einops import rearrange

try:
    import apex
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('MAE training')

    # dataset
    parser.add_argument('--root', type=str, required=True, help='root director of dataset')
    parser.add_argument('--image_size', type=int, default=(224, 224), nargs=2, help='input size')
    parser.add_argument('--crop', type=float, default=0.2, help='minimum crop')
    parser.add_argument('--aug', type=str, default='NULL', choices=['NULL', 'CJ', 'v1'],
                        help="augmentation type: NULL for normal supervised aug, CJ for aug with ColorJitter")
    parser.add_argument('--batch-size', type=int, default=64, help='batch_size per gpu')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers to use')

    # optimization
    parser.add_argument('--base-learning-rate', '--base-lr', type=float, default=0.1,
                        help='base learning when batch size = 256. final lr is determined by linear scale')
    parser.add_argument('--lr-scheduler', type=str, default='cosine', choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'lars', 'adamw'])
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr-decay-epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')

    # io
    parser.add_argument('--print-freq', type=int, default=20, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='save frequency')
    parser.add_argument('--output-dir', type=str, default='./output', help='output director')
    parser.add_argument('--postfix', type=str, default="")

    # misc
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng-seed", type=int, default=0, help='manual seed')

    args = parser.parse_args()

    args.model_name = 'MAE_transformer_bsz_{}x{}_ep{}'.format(args.batch_size, args.num_workers, args.epochs)
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)

    return args


def get_loader(args):
    train_transform = getTrans(args)
    train_dataset = MyDataSet(args.root, transform=train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    return train_loader


def build_model():
    img_size = 224
    patch_size = 16
    in_chans = 3
    encoder_num_classes = 0
    decoder_num_classes = 3 * 16 ** 2
    encoder_embed_dim = 1024
    decoder_embed_dim = 512
    encoder_depth = 8
    decoder_depth = 6
    encoder_heads = 8
    decoder_heads = 4
    mlp_ratio = 4
    drop_rate = 0
    attn_drop_rate = 0
    drop_path_rate = 0
    encoder_qkv_bias = True
    decoder_qkv_bias = False
    model = models.EncoderToDecoder(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                    encoder_num_classes=encoder_num_classes, decoder_num_classes=decoder_num_classes,
                                    encoder_embed_dim=encoder_embed_dim, decoder_embed_dim=decoder_embed_dim,
                                    encoder_depth=encoder_depth, decoder_depth=decoder_depth,
                                    encoder_heads=encoder_heads, decoder_heads=decoder_heads,
                                    mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                    encoder_qkv_bias=encoder_qkv_bias, decoder_qkv_bias=decoder_qkv_bias).cuda()
    return model


def load_checkpoint(args, model, optimizer, scheduler):
    logger.info("=> loading checkpoint '{}'".format(args.resume))

    checkpoint = torch.load(args.resume, map_location='cpu')
    args.start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    if args.amp_opt_level != "O0" and checkpoint['opt'].amp_opt_level != "O0":
        amp.load_state_dict(checkpoint['amp'])

    logger.info("=> loaded successfully '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(args, epoch, model, optimizer, scheduler):
    logger.info('==> Saving...')
    state = {'opt': args, 'model': model.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict()}
    if args.amp_opt_level != "O0":
        state['amp'] = amp.state_dict()

    files = glob.glob(os.path.join(args.output_dir, 'epoch*.pth'))
    files.sort()
    if len(files) > 1:
        for i in range(len(files) - 1):
            os.remove(files[i])
        files = glob.glob(os.path.join(args.output_dir, 'epoch*.pth'))
        assert len(files) == 1
    torch.save(state, os.path.join(args.output_dir, 'epoch_{}.pth'.format(epoch)))


def main(args):
    train_loader = get_loader(args)
    n_data = len(train_loader.dataset)
    logger.info(f"length of training dataset: {n_data}")
    model = build_model()

    criterion = nn.MSELoss(reduction='mean').cuda()
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), momentum=args.momentum, weight_decay=args.weight_decay,
                                    lr=args.batch_size * dist.get_world_size() / 256 * args.base_learning_rate)
    elif args.optim == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.batch_size * dist.get_world_size() / 256 * args.base_learning_rate)
    else:
        raise NotImplementedError('optim {} has bot been implemented......'.format(args.optim))

    scheduler = get_scheduler(optimizer, len(train_loader), args)
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False, find_unused_parameters=True)

    if dist.get_rank() == 0:
        save_checkpoint(args, 0, model, optimizer, scheduler)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)

        tic = time.time()
        train_mae(epoch, train_loader, model, criterion, optimizer, scheduler, args)

        use_time = time.time() - tic
        used_min = (epoch * use_time) / 60
        need_min = (args.epochs - epoch) * use_time / 60
        logger.info('epoch {}, total time {:.2f}, '.format(epoch, use_time))
        logger.info(
            'use {:d} hour {} min, need {} hour {:.4f} min yet.'.format(int(used_min // 60), used_min % 60, need_min // 60, need_min % 60))

        if dist.get_rank() == 0 and (epoch % 200 == 0 or epoch == 1):
            save_checkpoint(args, epoch, model, optimizer, scheduler)


def train_mae(epoch, train_loader, model, criterion, optimizer, scheduler, args, norm_target=False, patch_size=16):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    end = time.time()
    prefetcher = DataPrefetcher(train_loader)
    idx = 0
    inputs = prefetcher.next()

    while inputs is not None:
        data_time.update(time.time() - end)

        x, mask = inputs
        mask = mask.to(torch.bool)
        bsz = x.size(0)
        decoder_output = model(x, mask)  # [batch, mask_num, dim]

        with torch.no_grad():
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(x.device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(x.device)[None, :, None, None]
            unnorm_image = x * std + mean

            if norm_target:
                squeeze_images = rearrange(unnorm_image, "b c (h p1) (w p2) -> b (h w) (p1 p2) c", p1=patch_size, p2=patch_size)
                squeeze_norm_images = (squeeze_images - squeeze_images.mean(dim=-2, keepdim=True)) / (
                            squeeze_images.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                images_patch = rearrange(squeeze_norm_images, "b n p c -> b n (p c)")
            else:
                images_patch = rearrange(unnorm_image, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size)

            b, n, d = images_patch.size()
            labels = images_patch[mask].view(b, -1, d)

        loss = criterion(decoder_output, labels)

        # backward
        optimizer.zero_grad()
        if args.amp_opt_level != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

        # update meters
        loss_meter.update(loss.item(), bsz)
        batch_time.update(time.time() - end)
        end = time.time()
        lr = scheduler.get_lr()[0]

        # print info
        if idx % args.print_freq == 0:
            logger.info(f'Train: [{epoch}][{idx}/{len(train_loader)}]\tT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'DT {data_time.val:.3f} ({data_time.avg:.3f})\tlr: {lr:.6f}\tloss {loss_meter.val:.6f} ({loss_meter.avg:.6f})\t')

        inputs = prefetcher.next()
        idx += 1

    return loss_meter.avg,


if __name__ == '__main__':
    gpu_listener(os.environ['CUDA_VISIBLE_DEVICES'], gpu_total=8)
    opt = parse_option()
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    if dist.get_rank() == 0:
        os.makedirs(opt.output_dir, exist_ok=True)
    logger = setup_logger(output=opt.output_dir, distributed_rank=dist.get_rank(), name="MAE")
    if dist.get_rank() == 0:
        path = os.path.join(opt.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))
    main(opt)
