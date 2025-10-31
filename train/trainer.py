# -*- coding: utf-8 -*-
"""
trainer.py

This module defines the training, evaluation, and testing pipeline for the LPSGM (Large Polysomnography Model) project. 
It includes the loss function, learning rate scheduler with warmup, and the Trainer class that manages data loading, 
model training, checkpointing, and performance logging. The Trainer class supports resuming from checkpoints and 
tracks best evaluation and test metrics for model selection.

Key components:
- ClassifyLoss: Weighted cross-entropy loss tailored for sleep staging classes.
- WarmupCosineAnnealingLR: Learning rate scheduler combining warmup and cosine annealing.
- Trainer: Orchestrates the training loop, evaluation, testing, checkpoint saving/loading, and TensorBoard logging.

The module leverages PyTorch for model training, tensorboard for visualization, and tqdm for progress bars.
"""

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset.dataloader import ClassifyDataLoader
from model.model import LPSGM
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import os
from utils import *
import torch.nn.functional as F

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True


class ClassifyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Class weights for imbalanced sleep stage classification
        self.weight = torch.tensor([0.07903392, 0.47365554, 0.06655968, 0.19538417, 0.18536669])
            
    def forward(self, output, target):
        """
        Compute weighted cross-entropy loss for multi-class sleep stage classification.

        Args:
            output (torch.Tensor): Model predictions with shape (batch_size, sequence_length, 5)
            target (torch.Tensor): Ground truth labels with shape (batch_size, sequence_length)

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Move class weights to the same device as target
        self.weight = self.weight.to(target.device)
        # Permute output to shape (batch_size, 5, sequence_length) for cross_entropy
        output = output.permute(0, 2, 1)
        loss = F.cross_entropy(
            output, 
            target, 
            reduction='mean', 
            weight=self.weight
        )
        return loss


class WarmupCosineAnnealingLR(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, warmup_epoch, eta_min=0, last_epoch=-1, verbose=False):
        """
        Learning rate scheduler with linear warmup followed by cosine annealing.

        Args:
            optimizer (torch.optim.Optimizer): Wrapped optimizer.
            T_max (int): Maximum number of iterations.
            warmup_epoch (int): Number of warmup epochs.
            eta_min (float, optional): Minimum learning rate. Defaults to 0.
            last_epoch (int, optional): The index of last epoch. Defaults to -1.
            verbose (bool, optional): If True, prints a message to stdout on update. Defaults to False.
        """
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, T_max, eta_min, last_epoch, verbose)

    def get_lr(self):
        """
        Compute learning rate at current epoch with warmup and cosine annealing.

        Returns:
            list: List of learning rates for each parameter group.
        """
        if self.last_epoch < self.warmup_epoch:
            # Linear warmup scaling of learning rate
            return [base_lr * self.last_epoch / self.warmup_epoch for base_lr in self.base_lrs]
        # After warmup, use cosine annealing schedule
        return super().get_lr()


class Trainer:
    def __init__(self, args):
        """
        Initialize the Trainer with dataset, model, optimizer, scheduler, loss function, and logging.

        Args:
            args (Namespace): Configuration arguments containing dataset splits, hyperparameters, and paths.
        """
        self.args = args
        # Initialize data loader for training, evaluation, and testing datasets
        self.dataset = ClassifyDataLoader(
            args.train_subjects,
            args.eval_subjects,
            args.test_subjects,
            args.seq_len,
            args.batch_size,
            args.num_workers,
            args.num_processes,
            args.cache_root,
            args.random_shift_len,
        )
                
        # Initialize LPSGM model wrapped with DataParallel for multi-GPU support and move to GPU
        self.model = nn.DataParallel(LPSGM(args)).cuda()
        model_summary(self.model)   # Print model summary

        # Initialize AdamW optimizer with learning rate and weight decay
        self.optimizer = AdamW(self.model.parameters(),
                               lr=args.lr0, weight_decay=args.weight_decay)
        
        # Initialize learning rate scheduler with warmup and cosine annealing
        self.scheduler = WarmupCosineAnnealingLR(
            self.optimizer,
            T_max = args.epochs,
            warmup_epoch = args.warmup_epochs,
            eta_min = args.eta_min,
            verbose = True,
        )

        # Define weighted cross-entropy loss for classification
        self.loss_func = ClassifyLoss()

        # TensorBoard logger for visualization
        self.logger = SummaryWriter(args.log_dir)

        # Initialize training state variables
        self.n_epoch = 0 
        self.n_step = 0
        self.eval_acc_best = 0
        self.eval_f1_best = 0
        self.test_acc_best = 0
        self.test_f1_best = 0

        # Resume training from checkpoint if specified
        if args.state_dict_file != None:
            self.resume()
            

    def resume(self):
        """
        Resume training from a saved checkpoint including model, optimizer, scheduler states and metrics.
        """
        print(f'Resumings from pretrain weight file: {self.args.state_dict_file}')
        state_dict = torch.load(self.args.state_dict_file)

        # Load model weights
        self.model.load_state_dict(state_dict['model_state_dict'])
        # Load optimizer state
        self.optimizer.load_state_dict(state_dict['model_optimizer_state_dict'])
        # Load scheduler state
        self.scheduler.load_state_dict(state_dict['model_scheduler_state_dict'])

        # Restore training progress and best metrics
        self.n_epoch = state_dict['n_epoch'] + 1
        self.n_step = state_dict['n_step']
        self.eval_acc_best = state_dict['eval_acc']
        self.eval_f1_best = state_dict['eval_f1']
        self.test_acc_best = state_dict['test_acc']
        self.test_f1_best = state_dict['test_f1']

        # Evaluate model on test set after resuming
        self.test()
        

    def save_checkpoint(self, model_name):
        """
        Save the current training state including model, optimizer, scheduler, and metrics.

        Args:
            model_name (str): Filename for saving the checkpoint.
        """
        os.makedirs(self.args.model_dir, exist_ok=True)
        model_path = os.path.join(self.args.model_dir, model_name)
        torch.save({
            'n_epoch': self.n_epoch,
            'n_step': self.n_step,
            'eval_acc': self.eval_acc_best,
            'eval_f1': self.eval_f1_best,
            'test_acc': self.test_acc_best,
            'test_f1': self.test_f1_best,

            'model_state_dict': self.model.state_dict(),
            'model_optimizer_state_dict': self.optimizer.state_dict(),
            'model_scheduler_state_dict': self.scheduler.state_dict(),
            }, model_path)


    def train(self):
        """
        Execute the full training loop over the specified number of epochs, including training, evaluation, testing,
        learning rate scheduling, checkpointing, and logging.

        Returns:
            nn.Module: Trained model after completion of all epochs.
        """
        for _ in range(self.n_epoch, self.args.epochs):
            print(f'\n\nEpoch: {self.n_epoch}')
            # Get training data loader for current epoch
            train_data_loader = self.dataset.get_train_data_loader()
            print('Train data length:', len(train_data_loader))
            
            self.model.train()  # Set model to training mode
            
            losses = AverageMeter()  # Track average training loss

            for x, y, seq_idx, ch_idx, mask, ori_len in tqdm(train_data_loader):
                # x: input tensor with shape (batch_size, sequence_length * channels, 3000 samples)
                # y: target labels with shape (batch_size, sequence_length)
                # seq_idx, ch_idx, mask: auxiliary tensors with shape (batch_size, sequence_length * channels)

                # Move data to GPU
                x, y, seq_idx, ch_idx, mask = x.cuda(), y.cuda(), seq_idx.cuda(), ch_idx.cuda(), mask.cuda()

                self.optimizer.zero_grad()  # Reset gradients
                # Forward pass through model
                out = self.model(x, mask, ch_idx, seq_idx, ori_len)  # Output shape: (batch_size, sequence_length, 5)

                loss = self.loss_func(out, y)  # Compute classification loss
                loss.backward()  # Backpropagation

                # Gradient clipping to stabilize training if clip_value > 0
                if self.args.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_value)
                self.optimizer.step()  # Update model parameters

                # Update loss meter and log training loss per step
                losses.update(loss.item())
                self.logger.add_scalar('train/1.loss', 
                                       loss.item(), 
                                       self.n_step)

                self.n_step += 1
            
            # Log average training loss per epoch
            self.logger.add_scalar('train/2.loss',
                                   losses.avg,
                                   self.n_epoch)
            
            print("###Train Result###\nIter: {}/{}, loss: {:.4f}".format(
                self.n_epoch+1, self.args.epochs, losses.avg
            ))

            # Perform evaluation and testing after each epoch
            self.eval()
            self.test()
            
            # Update learning rate scheduler
            self.scheduler.step()
            self.n_epoch += 1

        # Close TensorBoard logger and clear dataset cache after training
        self.logger.close()
        self.dataset.clear_cache()

        return self.model
        

    @torch.no_grad()
    def test(self):
        """
        Evaluate the model on the test dataset, compute metrics, log results, and save checkpoint if performance improves.
        """
        preds, targets = [], []
        losses = AverageMeter()

        self.model.eval()  # Set model to evaluation mode

        test_data_loader = self.dataset.get_test_data_loader()
        print('Test data length:', len(test_data_loader))

        for x, y, seq_idx, ch_idx, mask, ori_len in tqdm(test_data_loader):
            # Move data to GPU
            x, y, seq_idx, ch_idx, mask = x.cuda(), y.cuda(), seq_idx.cuda(), ch_idx.cuda(), mask.cuda()

            # Forward pass
            out = self.model(x, mask, ch_idx, seq_idx, ori_len)  # Output shape: (batch_size, sequence_length, 5)
            loss = self.loss_func(out, y)  # Compute loss

            losses.update(loss.item())    

            # Convert logits to probabilities
            logits = torch.softmax(out, dim=-1) # Shape: (batch_size, sequence_length, 5)
            # Get predicted class indices
            _, pred = torch.max(logits, dim=-1) # Shape: (batch_size, sequence_length)

            # Accumulate predictions and targets for metric computation
            preds.extend( pred.cpu().numpy().flatten().tolist() )
            targets.extend( y.cpu().numpy().flatten().tolist() )
        
        # Compute evaluation metrics including accuracy, F1 scores, confusion matrix, and Cohen's kappa
        acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa = \
            get_metric(targets, preds)
        
        # Save checkpoint if test accuracy improves
        if acc > self.test_acc_best:
            self.test_acc_best = acc
            self.test_f1_best = f1
            self.save_checkpoint("test_best.pth")
            print("Test performance updated! acc: {:.4f}, f1: {:.4f}".format(
                self.test_acc_best, self.test_f1_best))

        # Log test metrics to TensorBoard
        self.logger.add_scalar('test/1.acc', acc, self.n_epoch)
        self.logger.add_scalar('test/2.f1', f1, self.n_epoch)
        self.logger.add_scalar('test/kappa', kappa, self.n_epoch)
        self.logger.add_scalar('test/4.loss', losses.avg, self.n_epoch)
        self.logger.add_scalars('test/5.classes_f1',
                                {'W': wake_f1,
                                 'N1': n1_f1,
                                 'N2': n2_f1,
                                 'N3': n3_f1,
                                 'R': rem_f1}, self.n_epoch)
        
        print("###Test Result###\nacc: {:.4f}, f1: {:.4f}, kappa: {:.4f}, loss: {:.4f}".format(
            acc, f1, kappa, losses.avg
        ))
        print("classes f1: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
            wake_f1, n1_f1, n2_f1, n3_f1, rem_f1
        ))


    @torch.no_grad()
    def eval(self):
        """
        Evaluate the model on the validation dataset, compute metrics, log results, and save checkpoint if performance improves.
        """
        preds, targets = [], []
        losses = AverageMeter()

        self.model.eval()  # Set model to evaluation mode

        eval_dataloader = self.dataset.get_eval_data_loader()
        print('Eval data length:', len(eval_dataloader))

        for x, y, seq_idx, ch_idx, mask, ori_len in tqdm(eval_dataloader):
            # Move data to GPU
            x, y, seq_idx, ch_idx, mask = x.cuda(), y.cuda(), seq_idx.cuda(), ch_idx.cuda(), mask.cuda()

            # Forward pass
            out = self.model(x, mask, ch_idx, seq_idx, ori_len)  # Output shape: (batch_size, sequence_length, 5)
            loss = self.loss_func(out, y)  # Compute loss

            losses.update(loss.item())    

            # Convert logits to probabilities
            logits = torch.softmax(out, dim=-1) # Shape: (batch_size, sequence_length, 5)
            # Get predicted class indices
            _, pred = torch.max(logits, dim=-1) # Shape: (batch_size, sequence_length)

            # Accumulate predictions and targets for metric computation
            preds.extend( pred.cpu().numpy().flatten().tolist() )
            targets.extend( y.cpu().numpy().flatten().tolist() )
        
        # Compute evaluation metrics including accuracy, F1 scores, confusion matrix, and Cohen's kappa
        acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa = \
            get_metric(targets, preds)
        
        # Save checkpoint if evaluation accuracy improves
        if acc > self.eval_acc_best:
            self.eval_acc_best = acc
            self.eval_f1_best = f1
            self.save_checkpoint("eval_best.pth")
            print("Eval erformance updated! acc: {:.4f}, f1: {:.4f}".format(
                self.eval_acc_best, self.eval_f1_best))
        
        # Save checkpoint for current epoch or as latest based on configuration
        if self.args.save_epoch:
            self.save_checkpoint(f"checkpoint-{self.n_epoch}.pth")
        else:
            self.save_checkpoint("latest.pth")
        
        # Log evaluation metrics to TensorBoard
        self.logger.add_scalar('eval/1.acc', acc, self.n_epoch)
        self.logger.add_scalar('eval/2.f1', f1, self.n_epoch)
        self.logger.add_scalar('eval/kappa', kappa, self.n_epoch)
        self.logger.add_scalar('eval/4.loss', losses.avg, self.n_epoch)
        self.logger.add_scalars('eval/5.classes_f1',
                                {'W': wake_f1,
                                 'N1': n1_f1,
                                 'N2': n2_f1,
                                 'N3': n3_f1,
                                 'R': rem_f1}, self.n_epoch)
        
        print("###Eval Result###\nacc: {:.4f}, f1: {:.4f}, kappa: {:.4f}, loss: {:.4f}".format(
            acc, f1, kappa, losses.avg
        ))
        print("classes f1: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
            wake_f1, n1_f1, n2_f1, n3_f1, rem_f1
        ))
