import sys
import os
from os import makedirs, remove
from os.path import exists, join
import logging
import re
import functools
import fnmatch
import numpy as np
import cv2
import h5py

import torch.optim.lr_scheduler as toptim
import tensorflow as tf
import numpy as np
import scipy.misc
try:
  from StringIO import StringIO  # Python 2.7
except ImportError:
  from io import BytesIO         # Python 3.x

def breakpoint():
    import pdb;pdb.set_trace()

def save_batch_nn(nn_name, pred_dict, input_batch, basename_list, save_dir):
    if not exists(save_dir):
        makedirs(save_dir)
    batch_size = pred_dict['partcls_per_point'].size(0)
    for b in range(batch_size):
        f = h5py.File(os.path.join(save_dir, basename_list[int(input_batch['index'][b])] + '.h5'), 'w')
        f.attrs['method_name'] = nn_name
        f.attrs['basename'] = basename_list[int(input_batch['index'][b])]
        for key, value in pred_dict.items():
            f.create_dataset('{}_pred'.format(key), data=value[b].cpu().numpy())
        for key, value in input_batch.items():
            f.create_dataset('{}_gt'.format(key), data=value[b].cpu().numpy())

    print('---saving to ', save_dir)

def decide_checkpoints(cfg, keys=['point'], suffix='valid'):
    for key in keys:
        if (cfg.use_pretrain or cfg.eval):
            if len(cfg.pretrained_path)>0:
                cfg[key].encoder_weights = cfg.pretrained_path + f'/encoder_best_{key}_{suffix}.pth'
                cfg[key].decoder_weights = cfg.pretrained_path + f'/encoder_best_{key}_{suffix}.pth'
                cfg[key].header_weights  = cfg.pretrained_path + f'/head_best_{key}_{suffix}.pth'
            else:
                cfg[key].encoder_weights = cfg.log_dir + f'/encoder_best_{key}_{suffix}.pth'
                cfg[key].decoder_weights = cfg.log_dir + f'/encoder_best_{key}_{suffix}.pth'
                cfg[key].header_weights  = cfg.log_dir + f'/head_best_{key}_{suffix}.pth'
        else:
            cfg[key].encoder_weights = ''
            cfg[key].decoder_weights = ''
            cfg[key].header_weights  = ''
        if cfg.verbose:
            print(cfg[key].encoder_weights)
            print(cfg[key].decoder_weights)
            print(cfg[key].header_weights)

def save_to_log(logdir, logger, info, epoch, img_summary=False, imgs=[]):
    # save scalars
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)

    if img_summary and len(imgs) > 0:
        directory = os.path.join(logdir, "predictions")
        if not os.path.isdir(directory):
            os.makedirs(directory)
        for i, img in enumerate(imgs):
            name = os.path.join(directory, str(i) + ".png")
            cv2.imwrite(name, img)

def make_log_img(depth, mask, pred, gt, color_fn):
    # input should be [depth, pred, gt]
    # make range image (normalized to 0,1 for saving)
    depth = (cv2.normalize(depth, None, alpha=0, beta=1,
                           norm_type=cv2.NORM_MINMAX,
                           dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
    out_img = cv2.applyColorMap(
        depth, get_mpl_colormap('viridis')) * mask[..., None]

    # make label prediction
    pred_color = color_fn((pred * mask).astype(np.int32))
    out_img = np.concatenate([out_img, pred_color], axis=0)
    # make label gt
    gt_color = color_fn(gt)
    out_img = np.concatenate([out_img, gt_color], axis=0)
    return (out_img).astype(np.uint8)

class Logger(object):

  def __init__(self, log_dir):
    """Create a summary writer logging to log_dir."""
    self.writer = tf.summary.FileWriter(log_dir)

  def scalar_summary(self, tag, value, step):
    """Log a scalar variable."""
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=tag, simple_value=value)])
    self.writer.add_summary(summary, step)
    self.writer.flush()

  def image_summary(self, tag, images, step):
    """Log a list of images."""

    img_summaries = []
    for i, img in enumerate(images):
      # Write the image to a string
      try:
        s = StringIO()
      except:
        s = BytesIO()
      scipy.misc.toimage(img).save(s, format="png")

      # Create an Image object
      img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                 height=img.shape[0],
                                 width=img.shape[1])
      # Create a Summary value
      img_summaries.append(tf.Summary.Value(
          tag='%s/%d' % (tag, i), image=img_sum))

    # Create and write Summary
    summary = tf.Summary(value=img_summaries)
    self.writer.add_summary(summary, step)
    self.writer.flush()

  def histo_summary(self, tag, values, step, bins=1000):
    """Log a histogram of the tensor of values."""

    # Create a histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins)

    # Fill the fields of the histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
      hist.bucket_limit.append(edge)
    for c in counts:
      hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    self.writer.add_summary(summary, step)
    self.writer.flush()

class warmupLR(toptim._LRScheduler):
  """ Warmup learning rate scheduler.
      Initially, increases the learning rate from 0 to the final value, in a
      certain number of steps. After this number of steps, each step decreases
      LR exponentially.
  """
  def __init__(self, optimizer, lr, warmup_steps, momentum, decay):
    # cyclic params
    self.optimizer = optimizer
    self.lr = lr
    self.warmup_steps = warmup_steps
    self.momentum = momentum
    self.decay = decay

    # cap to one
    if self.warmup_steps < 1:
      self.warmup_steps = 1

    # cyclic lr
    self.initial_scheduler = toptim.CyclicLR(self.optimizer,
                                             base_lr=0,
                                             max_lr=self.lr,
                                             step_size_up=self.warmup_steps,
                                             step_size_down=self.warmup_steps,
                                             cycle_momentum=False,
                                             base_momentum=self.momentum,
                                             max_momentum=self.momentum)

    # our params
    self.last_epoch = -1  # fix for pytorch 1.1 and below
    self.finished = False  # am i done
    super().__init__(optimizer)

  def get_lr(self):
    return [self.lr * (self.decay ** self.last_epoch) for lr in self.base_lrs]

  def step(self, epoch=None):
    if self.finished or self.initial_scheduler.last_epoch >= self.warmup_steps:
      if not self.finished:
        self.base_lrs = [self.lr for lr in self.base_lrs]
        self.finished = True
      return super(warmupLR, self).step(epoch)
    else:
      return self.initial_scheduler.step(epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)

def setup_logger(distributed_rank=0, filename="log.txt"):
    logger = logging.getLogger("Logger")
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    return logger


def find_recursive(root_dir, ext='.jpg'):
    files = []
    for root, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, '*' + ext):
            files.append(os.path.join(root, filename))
    return files


class NotSupportedCliException(Exception):
    pass


def process_range(xpu, inp):
    start, end = map(int, inp)
    if start > end:
        end, start = start, end
    return map(lambda x: '{}{}'.format(xpu, x), range(start, end+1))


REGEX = [
    (re.compile(r'^gpu(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^(\d+)$'), lambda x: ['gpu%s' % x[0]]),
    (re.compile(r'^gpu(\d+)-(?:gpu)?(\d+)$'),
     functools.partial(process_range, 'gpu')),
    (re.compile(r'^(\d+)-(\d+)$'),
     functools.partial(process_range, 'gpu')),
]


def parse_devices(input_devices):

    """Parse user's devices input str to standard format.
    e.g. [gpu0, gpu1, ...]

    """
    ret = []
    for d in input_devices.split(','):
        for regex, func in REGEX:
            m = regex.match(d.lower().strip())
            if m:
                tmp = func(m.groups())
                # prevent duplicate
                for x in tmp:
                    if x not in ret:
                        ret.append(x)
                break
        else:
            raise NotSupportedCliException(
                'Can not recognize device: "{}"'.format(d))
    return ret
