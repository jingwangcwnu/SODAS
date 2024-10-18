import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torchvision.datasets import ImageFolder

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from pytorchtools import EarlyStopping
from sklearn.metrics import cohen_kappa_score, roc_auc_score, f1_score, balanced_accuracy_score


parser = argparse.ArgumentParser("IDRiD")
parser.add_argument('--data', type=str, default='../data/IDRiD/test', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=300, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='SOD-NAS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

IDRiD_CLASSES = 5

class ImageFolderWithSkip(ImageFolder):
    def __getitem__(self, index):
        try:
            return super(ImageFolderWithSkip, self).__getitem__(index)
        except OSError:
            # If a file is corrupted, ignore it and continue reading the next file.
            print("OSError: skip a bad image")
            return self[index + 1]


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, IDRiD_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  train_transform, valid_transform = utils._data_transforms_IDRiD(args)
  train_data = ImageFolderWithSkip(root=args.data, transform=train_transform)
  valid_data = ImageFolderWithSkip(root=args.data, transform=valid_transform)

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

  for epoch in range(args.epochs):
    scheduler.step()
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    
    train_acc, train_obj, train_kappa, train_roc_auc, train_f1, train_iba = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)
    valid_acc, valid_obj, valid_kappa, valid_roc_auc, valid_f1, valid_iba= infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  kappa_values = utils.AvgrageMeter()
  roc_auc_values = utils.AvgrageMeter()
  f1_scores = utils.AvgrageMeter()
  iba_values = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)
     
    # Cohen's Kappa
    predicted_labels = np.argmax(logits.data.cpu().numpy(), axis=1)
    true_labels = target.data.cpu().numpy()
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    kappa_values.update(kappa, n)

    # ROC-AUC
    probas = nn.functional.softmax(logits, dim=1)
    roc_auc = None
    try: 
        roc_auc = roc_auc_score(true_labels, probas.data.cpu().numpy(), multi_class='ovo') 
        roc_auc_values.update(roc_auc, n)
    except ValueError: 
        pass 

    # F1-score
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    f1_scores.update(f1, n)

    # IBA
    iba = balanced_accuracy_score(true_labels, predicted_labels, adjusted=True)
    iba_values.update(iba, n)
    
    if roc_auc is not 'N/A':
        logging.info('train %03d %e %f %f %f %f %f %f', step, objs.avg, top1.avg, top5.avg, kappa, roc_auc, f1, iba)
    else:
        logging.info('train %03d %e %f %f %f %s %f %f', step, objs.avg, top1.avg, top5.avg, kappa, 'N/A', f1, iba)
             
  return top1.avg, objs.avg, kappa, roc_auc, f1, iba


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  kappa_values = utils.AvgrageMeter()
  roc_auc_values = utils.AvgrageMeter()
  f1_scores = utils.AvgrageMeter()
  iba_values = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data[0], n)
    top1.update(prec1.data[0], n)
    top5.update(prec5.data[0], n)

    # Cohen's Kappa
    predicted_labels = np.argmax(logits.data.cpu().numpy(), axis=1)
    true_labels = target.data.cpu().numpy()
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    kappa_values.update(kappa, n)

    # ROC-AUC
    probas = nn.functional.softmax(logits, dim=1)
    roc_auc = None
    try: 
        roc_auc = roc_auc_score(true_labels, probas.data.cpu().numpy(), multi_class='ovo') 
        roc_auc_values.update(roc_auc, n)
    except ValueError: 
        pass 

    # F1-score
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    f1_scores.update(f1, n)

    # IBA
    iba = balanced_accuracy_score(true_labels, predicted_labels, adjusted=True)
    iba_values.update(iba, n)

    if roc_auc is not 'N/A':
        logging.info('valid %03d %e %f %f %f %f %f %f', step, objs.avg, top1.avg, top5.avg, kappa, roc_auc, f1, iba)
    else:
        logging.info('valid %03d %e %f %f %f %s %f %f', step, objs.avg, top1.avg, top5.avg, kappa, 'N/A', f1, iba)
            
  return top1.avg, objs.avg, kappa, roc_auc, f1, iba


if __name__ == '__main__':
  main() 

