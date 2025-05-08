import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.dataloader import ClassifyDataLoader
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
        self.weight = torch.tensor([0.07903392, 0.47365554, 0.06655968, 0.19538417, 0.18536669])
            
    def forward(self, output, target):
        # output: (bz, seql, 5)
        # target: (bz, seql)
        self.weight = self.weight.to(target.device)
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
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, T_max, eta_min, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_epoch:
            return [base_lr * self.last_epoch / self.warmup_epoch for base_lr in self.base_lrs]
        return super().get_lr()


class Trainer:
    def __init__(self, args):
        self.args = args
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
                
        self.model = nn.DataParallel(LPSGM(args)).cuda()
        model_summary(self.model)   

        self.optimizer = AdamW(self.model.parameters(),
                               lr=args.lr0, weight_decay=args.weight_decay)
        
        self.scheduler = WarmupCosineAnnealingLR(
            self.optimizer,
            T_max = args.epochs,
            warmup_epoch = args.warmup_epochs,
            eta_min = args.eta_min,
            verbose = True,
        )

        self.loss_func = ClassifyLoss()

        self.logger = SummaryWriter(args.log_dir)

        self.n_epoch = 0 
        self.n_step = 0
        self.eval_acc_best = 0
        self.eval_f1_best = 0
        self.test_acc_best = 0
        self.test_f1_best = 0

        if args.state_dict_file != None:
            self.resume()
            

    def resume(self):
        print(f'Resumings from pretrain weight file: {self.args.state_dict_file}')
        state_dict = torch.load(self.args.state_dict_file)

        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['model_optimizer_state_dict'])
        self.scheduler.load_state_dict(state_dict['model_scheduler_state_dict'])

        self.n_epoch = state_dict['n_epoch'] + 1
        self.n_step = state_dict['n_step']
        self.eval_acc_best = state_dict['eval_acc']
        self.eval_f1_best = state_dict['eval_f1']
        self.test_acc_best = state_dict['test_acc']
        self.test_f1_best = state_dict['test_f1']

        self.test()
        

    def save_checkpoint(self, model_name):
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

        for _ in range(self.n_epoch, self.args.epochs):
            print(f'\n\nEpoch: {self.n_epoch}')
            train_data_loader = self.dataset.get_train_data_loader()
            print('Train data length:', len(train_data_loader))
            
            self.model.train()
            
            losses = AverageMeter()

            for x, y, seq_idx, ch_idx, mask, ori_len in tqdm(train_data_loader):
                # x: (bz, seql_cn, 3000)    
                # y: (bz, seql)
                # seq_idx, ch_idx, mask: (bz, seql_cn)

                x, y, seq_idx, ch_idx, mask = x.cuda(), y.cuda(), seq_idx.cuda(), ch_idx.cuda(), mask.cuda()

                self.optimizer.zero_grad()
                out = self.model(x, mask, ch_idx, seq_idx, ori_len)  # (bz, seql, 5)

                loss = self.loss_func(out, y)
                loss.backward()

                if self.args.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_value)
                self.optimizer.step()

                # logging
                losses.update(loss.item())
                self.logger.add_scalar('train/1.loss', 
                                       loss.item(), 
                                       self.n_step)

                self.n_step += 1
            
            self.logger.add_scalar('train/2.loss',
                                   losses.avg,
                                   self.n_epoch)
            
            print("###Train Result###\nIter: {}/{}, loss: {:.4f}".format(
                self.n_epoch+1, self.args.epochs, losses.avg
            ))

            # test
            self.eval()
            self.test()
            
            self.scheduler.step()
            self.n_epoch += 1

        # close and save logger
        self.logger.close()
        self.dataset.clear_cache()

        return self.model
        

    @torch.no_grad()
    def test(self):
        preds, targets = [], []
        losses = AverageMeter()

        self.model.eval()

        test_data_loader = self.dataset.get_test_data_loader()
        print('Test data length:', len(test_data_loader))

        for x, y, seq_idx, ch_idx, mask, ori_len in tqdm(test_data_loader):
            x, y, seq_idx, ch_idx, mask = x.cuda(), y.cuda(), seq_idx.cuda(), ch_idx.cuda(), mask.cuda()

            out = self.model(x, mask, ch_idx, seq_idx, ori_len)  # (bz, seql, 5)
            loss = self.loss_func(out, y)

            losses.update(loss.item())    

            logits = torch.softmax(out, dim=-1) # (bz, seql, 5)
            _, pred = torch.max(logits, dim=-1) # pred: (bz, seql)  y: (bz, seql)

            preds.extend( pred.cpu().numpy().flatten().tolist() )
            targets.extend( y.cpu().numpy().flatten().tolist() )
        
        acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa = \
            get_metric(targets, preds)
        
        if acc > self.test_acc_best:
            self.test_acc_best = acc
            self.test_f1_best = f1
            self.save_checkpoint("test_best.pth")
            print("Test performance updated! acc: {:.4f}, f1: {:.4f}".format(
                self.test_acc_best, self.test_f1_best))

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
        preds, targets = [], []
        losses = AverageMeter()

        self.model.eval()

        eval_dataloader = self.dataset.get_eval_data_loader()
        print('Eval data length:', len(eval_dataloader))

        for x, y, seq_idx, ch_idx, mask, ori_len in tqdm(eval_dataloader):
            x, y, seq_idx, ch_idx, mask = x.cuda(), y.cuda(), seq_idx.cuda(), ch_idx.cuda(), mask.cuda()

            out = self.model(x, mask, ch_idx, seq_idx, ori_len)  # (bz, seql, 5)
            loss = self.loss_func(out, y)

            losses.update(loss.item())    

            logits = torch.softmax(out, dim=-1) # (bz, seql, 5)
            _, pred = torch.max(logits, dim=-1) # pred: (bz, seql)  y: (bz, seql)

            preds.extend( pred.cpu().numpy().flatten().tolist() )
            targets.extend( y.cpu().numpy().flatten().tolist() )
        
        acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa = \
            get_metric(targets, preds)
        
        if acc > self.eval_acc_best:
            self.eval_acc_best = acc
            self.eval_f1_best = f1
            self.save_checkpoint("eval_best.pth")
            print("Eval erformance updated! acc: {:.4f}, f1: {:.4f}".format(
                self.eval_acc_best, self.eval_f1_best))
        
        if self.args.save_epoch:
            self.save_checkpoint(f"checkpoint-{self.n_epoch}.pth")
        else:
            self.save_checkpoint("latest.pth")
        
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


 