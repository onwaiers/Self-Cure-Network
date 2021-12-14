import math
import numpy as np
import torchvision.models as models
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os
import shutil
import torch
import datetime
import torch.nn as nn
from util import image_utils
import argparse
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model.resnet import resnet18
from util.plot_util import plot_confusion_matrix
from util.common_util import initialize_weight_goog
from util.raf_dataset_util import RAFDataSet
from util.common_util import AverageMeter, ProgressMeter, RecorderMeter

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/data/ljj/FER/RAF', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default='/data/ljj/project/RAN/checkpoint/ijba_res18_naive.pth.tar',
                        help='Pretrained weights')
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=10, help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--margin_1', type=float, default=0.15, help='Rank regularization margin. Details described in the paper.')
    parser.add_argument('--margin_2', type=float, default=0.2, help='Relabeling margin. Details described in the paper.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=70, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
    parser.add_argument('--cm_path', type=str, default='./log/' + time_str + 'cm.png')        
    parser.add_argument('--best_cm_path', type=str, default='./log/' + time_str + 'best_cm.png')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/' + time_str + 'model.pth')
    parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoints/'+time_str+'model_best.pth')
    return parser.parse_args()
    
     
args = parse_args()
print('--------args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------args----------\n')
best_acc = 0
        
def run_training():

    imagenet_pretrained = False
    res18 = resnet18(pretrained = imagenet_pretrained, drop_rate = args.drop_rate) 
    
    # init weight
    if not imagenet_pretrained:
         for m in res18.modules():
            initialize_weight_goog(m)

    # load weights of the pretrained model       
    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained) 
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['state_dict']
        model_state_dict = res18.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            if  ((key=='module.fc.weight')|(key=='module.fc.bias')):
                pass
            else:    
                new_key = key[7:]
                model_state_dict[new_key] = pretrained_state_dict[key]
                total_keys+=1
                if key in model_state_dict:
                    loaded_keys+=1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        res18.load_state_dict(model_state_dict, strict = False)  
    
    # data_transforms
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25))])
    
    train_dataset = RAFDataSet(args.raf_path, phase = 'train', transform = data_transforms, basic_aug = True)    
    
    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])                                           
    val_dataset = RAFDataSet(args.raf_path, phase = 'test', transform = data_transforms_val)    
    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)
    
    params = res18.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params,weight_decay = 1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay = 1e-4)
    else:
        raise ValueError("Optimizer not supported.")
    
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    recorder = RecorderMeter(args.epochs)
    res18 = res18.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    
    # margin_1 = args.margin_1
    # margin_2 = args.margin_2
    # beta = args.beta

    # 记录训练时的参数值
    txt_name = './log/' + time_str + 'log.txt'
    with open(txt_name, 'a') as f:
        f.write('--------args----------\n')
        for k in list(vars(args).keys()):
            f.write('%s: %s\n' % (k, vars(args)[k]))
        f.write('--------args----------\n')  

    for i in range(0, args.epochs):
        train_acc, train_los = train(train_loader, res18, criterion, optimizer, scheduler, i)
        val_acc, val_los = validate(val_loader, res18, criterion, optimizer, i)

        recorder.update(i, train_los, train_acc, val_los, val_acc)
        curve_name = time_str + 'cnn.png'
        recorder.plot_curve(os.path.join('./log/', curve_name))

        print('Current best accuracy: ', best_acc)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc) + '\n')
        
     
def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    running_loss = 0.0
    correct_sum = 0
    iter_cnt = 0
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             time_str,
                             prefix="Epoch: [{}]".format(epoch))
    margin_1 = args.margin_1
    margin_2 = args.margin_2
    beta = args.beta

    model.train()
    for batch_i, (imgs, targets, indexes) in enumerate(train_loader):
        batch_sz = imgs.size(0) 
        iter_cnt += 1
        tops = int(batch_sz * beta)
        optimizer.zero_grad()
        imgs = imgs.cuda()
        attention_weights, outputs = model(imgs)
        
        # Rank Regularization
        _, top_idx = torch.topk(attention_weights.squeeze(), tops)
        _, down_idx = torch.topk(attention_weights.squeeze(), batch_sz - tops, largest = False)

        high_group = attention_weights[top_idx]
        low_group = attention_weights[down_idx]
        high_mean = torch.mean(high_group)
        low_mean = torch.mean(low_group)
        # diff  = margin_1 - (high_mean - low_mean)
        diff  = low_mean - high_mean + margin_1

        if diff > 0:
            RR_loss = diff
        else:
            RR_loss = 0.0
        
        targets = targets.cuda()
        loss = criterion(outputs, targets) + RR_loss 
        loss.backward()
        optimizer.step()
        
        running_loss += loss
        _, predicts = torch.max(outputs, 1)
        correct_num = torch.eq(predicts, targets).sum()
        correct_sum += correct_num

        prec1 = accuracy(outputs.data, targets, topk=(1,))
        losses.update(loss.item(), imgs.size(0))
        top1.update(prec1[0].item(), imgs.size(0))

        if batch_i % args.print_freq == 0:
            progress.display(batch_i)


        # Relabel samples
        if epoch >= args.relabel_epoch:
            sm = torch.softmax(outputs, dim = 1)
            Pmax, predicted_labels = torch.max(sm, 1) # predictions
            Pgt = torch.gather(sm, 1, targets.view(-1,1)).squeeze() # retrieve predicted probabilities of targets
            true_or_false = Pmax - Pgt > margin_2
            update_idx = true_or_false.nonzero().squeeze() # get samples' index in this mini-batch where (Pmax - Pgt > margin_2)
            label_idx = indexes[update_idx] # get samples' index in train_loader
            relabels = predicted_labels[update_idx] # predictions where (Pmax - Pgt > margin_2)
            train_loader.dataset.label[label_idx.cpu().numpy()] = relabels.cpu().numpy() # relabel samples in train_loader
            
    scheduler.step()
    acc = correct_sum.float() / float(train_loader.dataset.__len__())
    running_loss = running_loss/iter_cnt
    print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (epoch, acc, running_loss))
    txt_name = './log/' + time_str + 'log.txt'
    with open(txt_name, 'a') as f:
        f.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f\n' % (epoch, acc, running_loss))
    return top1.avg, losses.avg

def validate(val_loader, model, criterion, optimizer, epoch):
    global best_acc
    with torch.no_grad():
        losses = AverageMeter('Loss', ':.4f')
        top1 = AverageMeter('Accuracy', ':6.3f')
        progress = ProgressMeter(len(val_loader),
                                [losses, top1],
                                time_str,
                                prefix='Test: ')
        class_names = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Anger', 'Neutral']

        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        model.eval()
        for batch_i, (imgs, targets, _) in enumerate(val_loader):
            _, outputs = model(imgs.cuda())
            targets = targets.cuda()
            loss = criterion(outputs, targets)
            running_loss += loss
            iter_cnt+=1
            _, predicts = torch.max(outputs, 1)
            correct_num  = torch.eq(predicts,targets)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += outputs.size(0)

            prec1 = accuracy(outputs.data, targets, topk=(1,))
            losses.update(loss.item(), imgs.size(0))
            top1.update(prec1[0].item(), imgs.size(0))
            
            _, predicted = torch.max(outputs.data, 1)
            if batch_i == 0:
                all_predicted = predicted
                all_targets = targets
            else:
                all_predicted = torch.cat((all_predicted, predicted),0)
                all_targets = torch.cat((all_targets, targets),0)
            
            if batch_i % args.print_freq == 0:
                progress.display(batch_i)

        running_loss = running_loss/iter_cnt   
        acc = bingo_cnt.float()/float(sample_cnt)
        acc = np.around(acc.numpy(),4)
        print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (epoch, acc, running_loss))

        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f: 
           f.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f\n" % (epoch, acc, running_loss)) 

        is_best = acc > best_acc
        if acc > best_acc:
            best_acc = acc


        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict()},
                        is_best, args)

        # Compute confusion matrix
        matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
        np.set_printoptions(precision=2)

        # Plot normalized confusion matrix
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                            title= ' Confusion Matrix (Accuracy: %.3f%%)'%(top1.avg))
        plt.savefig(args.cm_path)
        if is_best:
            shutil.copyfile(args.cm_path, args.best_cm_path)
        plt.close()

        return top1.avg, losses.avg

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint_path)
    if is_best:
        shutil.copyfile(args.checkpoint_path, args.best_checkpoint_path)
if __name__ == "__main__":                    
    run_training()
