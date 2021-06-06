import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import os.path
import time
import logging
import sys
import copy
import sklearn.metrics
import scipy.special
import pickle

from tqdm import tqdm
from torchvision import transforms, utils, io
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

from models import BGSplittingModel
from datasets.auxiliary_dataset import AuxiliaryDataset
from datasets.inaturalist17_dataset import iNaturalist17Dataset
from warmup_scheduler import GradualWarmupScheduler
from util import EMA

logger = logging.getLogger("bgsplit")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

@dataclass
class ModelArgs:
    num_foreground_categories: int
    num_auxiliary_categories: int
    aux_weight: float
    aux_labels_type: str
    bg_thresh: Optional[float]

    initial_lr: float
    end_lr: float
    momentum: float
    weight_decay: float
    warmup_epochs: int
    warmup_lr: float
    max_epochs: int

    model_dir: str
    log_dir: str
    use_cuda: bool
    val_frequency: int
    checkpoint_frequency: int

    resume_from: Optional[str]
    resume_training: bool



class TrainingLoop:
    def __init__(
            self,
            model_args: ModelArgs,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader):
        '''The training loop for background splitting models.'''
        self.model_args = model_args
        self.val_frequency = model_args.val_frequency
        self.checkpoint_frequency = model_args.checkpoint_frequency
        self.use_cuda = model_args.use_cuda
        self.model_dir = model_args.model_dir
        self.aux_weight = model_args.aux_weight
        self.writer = SummaryWriter(log_dir=model_args.log_dir)

        # Setup dataset
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Setup model
        self._setup_model()

        # Resume if requested
        resume_from = model_args.resume_from
        if resume_from:
            resume_training = model_args.resume_training
            self.load_checkpoint(resume_from, resume_training=resume_training)

        # Variables for estimating run-time
        self.train_batch_time = EMA(0)
        self.val_batch_time = EMA(0)
        self.train_batches_per_epoch = (
            len(self.train_dataloader.dataset) /
            self.train_dataloader.batch_size)
        self.val_batches_per_epoch = (
            len(self.val_dataloader.dataset) /
            self.val_dataloader.batch_size)
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.train_epoch_loss = 0
        self.train_epoch_main_loss = 0
        self.train_epoch_aux_loss = 0

    def _setup_model(self):
        num_foreground_categories = self.model_args.num_foreground_categories
        num_aux_classes = self.model_args.num_auxiliary_categories

        self.model = BGSplittingModel(
            num_main_classes=num_foreground_categories,
            num_aux_classes=num_aux_classes,
            fixed_bg_threshold=self.model_args.bg_thresh)

        if self.model_args.aux_labels_type == "imagenet":
            # Initialize auxiliary head to imagenet fc
            self.model.auxiliary_head.weight = self.model.backbone.fc.weight
            self.model.auxiliary_head.bias = self.model.backbone.fc.bias

        if self.use_cuda:
            self.model = self.model.cuda()

        self.model = nn.DataParallel(self.model)
        self.main_loss = nn.CrossEntropyLoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()
        self.start_epoch = 0
        self.end_epoch = self.model_args.max_epochs
        self.current_epoch = 0
        self.global_train_batch_idx = 0
        self.global_val_batch_idx = 0

        lr = self.model_args.initial_lr
        endlr = self.model_args.end_lr
        optim_params = dict(
            lr=lr,
            momentum=self.model_args.momentum,
            weight_decay=self.model_args.weight_decay,
        )
        self.optimizer = optim.SGD(self.model.parameters(), **optim_params)
        warmup_epochs = self.model_args.warmup_epochs
        if False:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, self.model_args.max_epochs - warmup_epochs,
                eta_min=endlr)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[20, 40, 60, 80], gamma=0.1)
        self.optimizer_scheduler = GradualWarmupScheduler(
            optimizer=self.optimizer,
            multiplier=1.0,
            warmup_epochs=warmup_epochs,
            initial_lr=args.warmup_lr,
            after_scheduler=scheduler)

    def _estimate_time_left(self):
        epochs_left = self.end_epoch - self.current_epoch - 1
        num_train_batches_left = (
            epochs_left * self.train_batches_per_epoch +
            max(0, self.train_batches_per_epoch - self.train_batch_idx - 1)
        )
        num_val_batches_left = (
            (1 + round(epochs_left / self.val_frequency)) * self.val_batches_per_epoch +
            max(0, self.val_batches_per_epoch - self.val_batch_idx - 1)
        )
        time_left = (
            num_train_batches_left * self.train_batch_time.value +
            num_val_batches_left * self.val_batch_time.value)
        return time_left

    def load_checkpoint(self, path: str, resume_training: bool=False):
        checkpoint_state = torch.load(path)
        self.model.load_state_dict(checkpoint_state['state_dict'])
        if resume_training:
            self.global_train_batch_idx = checkpoint_state['global_train_batch_idx']
            self.global_val_batch_idx = checkpoint_state['global_val_batch_idx']
            self.start_epoch = checkpoint_state['epoch'] + 1
            self.current_epoch = self.start_epoch
            self.optimizer.load_state_dict(
                checkpoint_state['optimizer'])
            self.optimizer_scheduler.load_state_dict(
                checkpoint_state['optimizer_scheduler'])

    def save_checkpoint(self, epoch, checkpoint_path: str):
        state = dict(
            global_train_batch_idx=self.global_train_batch_idx,
            global_val_batch_idx=self.global_val_batch_idx,
            model_args=self.model_args,
            epoch=epoch,
            state_dict=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            optimizer_scheduler=self.optimizer_scheduler.state_dict(),
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(state, checkpoint_path)

    def _validate(self, dataloader):
        self.model.eval()
        loss_value = 0
        main_gts = []
        aux_gts = []
        main_preds = []
        aux_preds = []
        for batch_idx, (images, main_labels, aux_labels) in enumerate(
                dataloader):
            batch_start = time.perf_counter()
            self.val_batch_idx = batch_idx
            if self.use_cuda:
                images = images.cuda()
                main_labels = main_labels.cuda()
                aux_labels = aux_labels.cuda()
            main_logits, aux_logits = self.model(images)
            valid_main_labels = main_labels != -1
            valid_aux_labels = aux_labels != -1
            main_loss_value = self.main_loss(
                main_logits[valid_main_labels],
                main_labels[valid_main_labels])
            aux_loss_value = self.aux_weight * self.auxiliary_loss(
                aux_logits[valid_aux_labels],
                aux_labels[valid_aux_labels])

            loss_value = torch.zeros_like(main_loss_value)
            if valid_main_labels.sum() > 0:
                loss_value += main_loss_value
            if valid_aux_labels.sum() > 0:
                loss_value += aux_loss_value
            loss_value = loss_value.item()

            main_pred = F.softmax(main_logits[valid_main_labels], dim=1)
            aux_pred = F.softmax(main_logits[valid_main_labels], dim=1)
            main_preds += list(main_pred.argmax(dim=1)[valid_main_labels].cpu().numpy())
            aux_preds += list(aux_pred.argmax(dim=1)[valid_aux_labels].cpu().numpy())
            main_gts += list(main_labels[valid_main_labels].cpu().numpy())
            aux_gts += list(aux_labels[valid_aux_labels].cpu().numpy())
            batch_end = time.perf_counter()
            self.val_batch_time += (batch_end - batch_start)
            self.global_val_batch_idx += 1
        # Compute F1 score
        if len(dataloader) > 0:
            loss_value /= (len(dataloader) + 1e-10)
            main_prec, main_recall, main_f1, _ = \
                sklearn.metrics.precision_recall_fscore_support(
                    main_gts, main_preds, average='macro',
                    labels=range(1, 1 + self.model_args.num_foreground_categories),
                    zero_division=0)
            aux_prec, aux_recall, aux_f1, _ = \
                sklearn.metrics.precision_recall_fscore_support(
                    aux_gts, aux_preds, average='macro',
                    zero_division=0)
        else:
            loss_value = 0
            main_prec = -1
            main_recall = -1
            main_f1 = -1
            aux_prec = -1
            aux_recall = -1
            aux_f1 = -1

        summary_data = [
            ('loss', loss_value),
            ('f1/main_head', main_f1),
            ('prec/main_head', main_prec),
            ('recall/main_head', main_recall),
            ('f1/aux_head', aux_f1),
            ('prec/aux_head', aux_prec),
            ('recall/aux_head', aux_recall),
        ]
        for k, v in [('val/epoch/' + tag, v) for tag, v in summary_data]:
            self.writer.add_scalar(k, v, self.current_epoch)

    def validate(self):
        self._validate(self.val_dataloader)

    def train(self):
        self.model.train()
        logger.info('Starting train epoch')
        load_start = time.perf_counter()
        self.train_epoch_loss = 0
        self.train_epoch_main_loss = 0
        self.train_epoch_aux_loss = 0
        main_gts = []
        aux_gts = []
        main_logits_all = []
        main_preds = []
        aux_preds = []
        total_batches = len(self.train_dataloader)
        for batch_idx, (images, main_labels, aux_labels) in tqdm(enumerate(
                self.train_dataloader), total=total_batches):
            load_end = time.perf_counter()
            batch_start = time.perf_counter()
            self.train_batch_idx = batch_idx
            if False:
                logger.debug('Train batch')
            if self.use_cuda:
                images = images.cuda()
                main_labels = main_labels.cuda()
                aux_labels = aux_labels.cuda()

            main_logits, aux_logits = self.model(images)
            # Compute loss
            valid_main_labels = main_labels != -1
            valid_aux_labels = aux_labels != -1
            main_loss_value = self.main_loss(
                main_logits[valid_main_labels],
                main_labels[valid_main_labels])
            aux_loss_value = self.aux_weight * self.auxiliary_loss(
                aux_logits[valid_aux_labels],
                aux_labels[valid_aux_labels])

            loss_value = torch.zeros_like(main_loss_value)
            if valid_main_labels.sum() > 0:
                loss_value += main_loss_value
            if valid_aux_labels.sum() > 0:
                loss_value += aux_loss_value

            self.train_epoch_loss += loss_value.item()
            if torch.sum(valid_main_labels) > 0:
                self.train_epoch_main_loss += main_loss_value.item()
            if torch.sum(valid_aux_labels) > 0:
                self.train_epoch_aux_loss += aux_loss_value.item()
            # Update gradients
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            model_lr = self.optimizer.param_groups[-1]['lr']
            self.optimizer_scheduler.step(
                epoch=self.current_epoch + batch_idx / total_batches)

            main_pred = F.softmax(main_logits[valid_main_labels], dim=1)
            aux_pred = F.softmax(aux_logits[valid_aux_labels], dim=1)
            main_logits_all += list(main_logits[valid_main_labels].detach().cpu().numpy())
            main_preds += list(main_pred.argmax(dim=1)[valid_main_labels].cpu().numpy())
            aux_preds += list(aux_pred.argmax(dim=1)[valid_aux_labels].cpu().numpy())
            main_gts += list(main_labels[valid_main_labels].cpu().numpy())
            aux_gts += list(aux_labels[valid_aux_labels].cpu().numpy())

            batch_end = time.perf_counter()
            total_batch_time = (batch_end - batch_start)
            total_load_time = (load_end - load_start)
            self.train_batch_time += total_batch_time + total_load_time
            if False:
                logger.debug(f'Train batch time: {self.train_batch_time.value}, '
                             f'this batch time: {total_batch_time}, '
                             f'this load time: {total_load_time}, '
                             f'batch epoch loss: {loss_value.item()}, '
                             f'main loss: {main_loss_value.item()}, '
                             f'aux loss: {aux_loss_value.item()}')
            summary_data = [
                ('lr', model_lr),
                ('loss', loss_value.item()),
                ('loss/main_head', main_loss_value.item()),
                ('loss/aux_head', aux_loss_value.item()),
            ]
            for k, v in [('train/batch/' + tag, v) for tag, v in summary_data]:
                self.writer.add_scalar(k, v, self.global_train_batch_idx)

            self.global_train_batch_idx += 1
            load_start = time.perf_counter()

        logger.debug(f'Train epoch loss: {self.train_epoch_loss}, '
                     f'main loss: {self.train_epoch_main_loss}, '
                     f'aux loss: {self.train_epoch_aux_loss}')
        main_prec, main_recall, main_f1, _ = \
            sklearn.metrics.precision_recall_fscore_support(
                main_gts, main_preds, average='macro',
                labels=range(1, 1 + self.model_args.num_foreground_categories),
                zero_division=0)
        aux_prec, aux_recall, aux_f1, _ = \
            sklearn.metrics.precision_recall_fscore_support(
                aux_gts, aux_preds, average='macro',
                zero_division=0)
        logger.debug(f'Train epoch main: {main_prec}, {main_recall}, {main_f1}, '
                     f'aux: {aux_prec}, {aux_recall}, {aux_f1}'
                     f'main loss: {self.train_epoch_main_loss}, '
                     f'aux loss: {self.train_epoch_aux_loss}')
        summary_data = [
            ('loss', self.train_epoch_loss),
            ('loss/main_head', self.train_epoch_main_loss),
            ('loss/aux_head', self.train_epoch_aux_loss),
            ('f1/main_head', main_f1),
            ('prec/main_head', main_prec),
            ('recall/main_head', main_recall),
            ('f1/aux_head', aux_f1),
            ('prec/aux_head', aux_prec),
            ('recall/aux_head', aux_recall)
        ]
        for k, v in [('train/epoch/' + tag , v) for tag, v in summary_data]:
            self.writer.add_scalar(k, v, self.current_epoch)

        self.writer.add_histogram(
            'train/epoch/softmax/main_head',
            scipy.special.softmax(main_logits_all, axis=1)[:, 1])

    def run(self):
        self.last_checkpoint_path = None
        for i in range(self.start_epoch, self.end_epoch):
            logger.info(f'Train: Epoch {i}')
            self.current_epoch = i
            self.train()
            if i % self.val_frequency == 0 or i == self.end_epoch - 1:
                logger.info(f'Validate: Epoch {i}')
                self.validate()
            if i % self.checkpoint_frequency == 0 or i == self.end_epoch - 1:
                logger.info(f'Checkpoint: Epoch {i}')
                self.last_checkpoint_path = os.path.join(
                    self.model_dir, f'checkpoint_{i:03}.pth')
                self.save_checkpoint(i, self.last_checkpoint_path)
        return self.last_checkpoint_path


def generate_aux_labels(use_cuda, dataloader, output_path):
    model = resnet.resnet50(pretrained=True, progress=False)
    if use_cuda:
        model = model.cuda(0)
    model = nn.DataParallel(model)
    model.eval()

    all_preds = []
    paths = dataloader.dataset.base_dataset.paths
    for batch_idx, (images, main_labels, aux_labels) in tqdm(enumerate(
            dataloader), total=len(dataloader)):
        if use_cuda:
            images = images.cuda()
            main_labels = main_labels.cuda()
            aux_labels = aux_labels.cuda()
        logits = model(images)
        preds = F.softmax(logits, dim=1).argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    aux_labels = {os.path.basename(p): v
                  for p, v in zip(paths, all_preds)}
    with open(output_path, 'wb') as f:
        pickle.dump(aux_labels, f)


def main(args):
    ## Setup dataset
    image_input_size = args.image_input_size
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    resize_size = int(image_input_size * 1.15)
    resize_size += int(resize_size % 2)
    val_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_input_size),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    if args.dataset_type == 'inaturalist17':
        train_dataset = iNaturalist17Dataset(
            root=args.data_root,
            ann_path=os.path.join(args.data_root, 'train2017.json'),
            transform=train_transform
        )
        val_dataset = iNaturalist17Dataset(
            root=args.data_root,
            ann_path=os.path.join(args.data_root, 'val2017.json'),
            transform=train_transform
        )
        # Read aux label
        alt = args.aux_labels_type
        if alt == 'imagenet':
            aux_labels_path = \
                './resnet50_labels.pickle'
                #'/raid/apoms/datasets/inaturalist/resnet50_labels.pickle'
            num_aux_classes = 1000
        else:
            exit(-1)

    else:
        exit(-1)

    if not args.generate_aux_labels:
        auxiliary_labels = {
            os.path.basename(k): v
            for k,v in np.load(aux_labels_path, allow_pickle=True).items()
        }
    else:
        auxiliary_labels = {}

    # Read split file
    with open(args.data_split, 'r') as f:
        foreground_categories = [int(x) for x in f.readlines()]
    train_dataloader = DataLoader(
        AuxiliaryDataset(
            dataset=train_dataset,
            foreground_categories=foreground_categories,
            auxiliary_labels=auxiliary_labels,
            labeled_subset=args.labeled_subset,
            restrict_aux_labels=not args.unrestrict_aux_labels),
        batch_size=args.batch_size,
        shuffle=True if not args.generate_aux_labels else False,
        num_workers=args.workers)
    val_dataloader = DataLoader(
        AuxiliaryDataset(
            dataset=val_dataset,
            foreground_categories=foreground_categories,
            auxiliary_labels=None,
            restrict_aux_labels=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers)

    if args.generate_aux_labels:
        generate_aux_labels(True, train_dataloader, 'resnet50_labels.pickle')
        return

    ## Setup training loop
    model_args = ModelArgs(
        num_foreground_categories=len(foreground_categories),
        num_auxiliary_categories=num_aux_classes,
        aux_weight=args.aux_weight,
        aux_labels_type=args.aux_labels_type,
        bg_thresh=args.bg_thresh,

        initial_lr=args.initial_lr,
        end_lr=args.final_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
        max_epochs=args.epochs,

        model_dir=args.output_dir,
        log_dir=os.path.join(args.output_dir, 'tensorboard'),
        use_cuda=True,
        val_frequency=args.val_frequency,
        checkpoint_frequency=args.checkpoint_frequency,

        resume_from=args.resume_from,
        resume_training=args.resume_training,
    )
    loop = TrainingLoop(
        model_args=model_args,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
    loop.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train background splitting models.")

    ## Output args
    parser.add_argument("--output-dir", type=str, default="./outputs",
                        help="directory to write checkpoints and logs")

    ## Data args
    parser.add_argument("--dataset-type", type=str, default="inaturalist17",
                        help="one of: inaturalist17, places365")
    parser.add_argument("--data-split", type=str, default="path/to/split/file.txt",
                        help="path to split file containing foreground category ids")
    parser.add_argument("--data-root", type=str, default="/path/to/data/root",
                        help="path to dataset root")
    parser.add_argument("--image-input-size", type=int, default=224,
                        help="input size to the network")

    ## Background splitting args
    parser.add_argument("--aux-labels-type", default="imagenet", type=str,
                        help="weight for the auxiliary loss")
    parser.add_argument("--aux-weight", default=0.1, type=float,
                        help="weight for the auxiliary loss")
    parser.add_argument("--bg-thresh", default=None, type=float,
                        help="background threshold (unused if None)")

    ## Subset args
    parser.add_argument("--labeled-subset", default=None, type=float,
                        help="percentage of dataset to train with")
    parser.add_argument("--unrestrict-aux-labels",
                        action='store_true',
                        help=("use aux labels for all data instead of labeled "
                              "subset"))

    ## Optimization args
    parser.add_argument("--epochs", default=90, type=int,
                        help="number of total epochs to train for")
    parser.add_argument("--batch-size", default=256, type=int,
                        help="batch size across all gpus")
    parser.add_argument("--initial-lr", default=0.01, type=float,
                        help="initial learning rate")
    parser.add_argument("--final-lr", type=float, default=0,
                        help="final learning rate")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="momentum")
    parser.add_argument("--weight-decay", default=1e-4, type=float,
                        help="weight decay")
    parser.add_argument("--warmup-epochs", default=1, type=int,
                        help="number of warmup epochs")
    parser.add_argument("--warmup-lr", default=0.001, type=float,
                        help="initial warmup learning rate")

    ## Misc args
    parser.add_argument("--workers", default=32, type=int,
                        help="number of data loading workers")
    parser.add_argument("--checkpoint-frequency", type=int, default=5,
                        help="Save the model periodically")
    parser.add_argument("--val-frequency", type=int, default=5,
                        help="Validate the model periodically")
    parser.add_argument("--seed", type=int, default=1234, help="seed")

    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--resume-training", action='store_true',
                        help="resume from model checkpoint")

    ## Generate aux labels
    parser.add_argument("--generate-aux-labels", action='store_true',
                        help="generate aux labels to use")

    args = parser.parse_args()
    main(args)
