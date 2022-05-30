import argparse
import math
import time
import resource
import torch
import torch.nn as nn
from models import FC,myNet
import numpy as np
import importlib
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import torch.optim as optim
from utils import *
import Optim
import os
from torch.utils.tensorboard import SummaryWriter   
from torch.utils.tensorboard import FileWriter
from metrics import *
#python main.py --gpu 0 --data data/log.csv --save save/out.pt --hidCNN 50 --L1Loss False --output_fun None
def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size,figs_fold = None, test_length = None, epoch = None):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None
        
        for X, Y in data.get_batches(X, Y, batch_size, False):
            output = model(X)
            if output.shape[0] != Y.shape[0]:
                break
            if predict is None:
                predict = output
                test = Y
            else:
                predict = torch.cat((predict,output))
                test = torch.cat((test, Y))
        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()
        rse, corr, mae, mse, rmse, mape, mspe = metric(predict, Ytest)
        # print('rse:{},corr:{},mae:{},mse:{},rmse:{}.'.format(rse,corr,mae,mse,rmse))

        x = np.linspace(0,test_length,test_length)  #设置横轴的取值点
        columns = data.columns
        if figs_fold:
            fig_path = figs_fold+str(epoch)
            os.mkdir(fig_path)
            # columns = ['RESP_TIME']
            if data.pre_length > 1:
                for j in range(1):
                    for i in range(predict.shape[2]):
                        p1, = plt.plot(x,Ytest[:test_length,j,i],color='blue',linewidth=1,label='GT')
                        p2, = plt.plot(x,(predict[:test_length,j,i]),color='red',linewidth=1,label='Predict')
                        plt.xlabel("mins/time")
                        plt.ylabel(columns[i])
                        plt.legend([p2, p1], ["Predict", "GT"], loc='upper left')
                        plt.savefig(fig_path+'/'+columns[i]+'_'+str(epoch)+'.png')
                        plt.close('all')
            if data.pre_length == 1:
                for i in range(predict.shape[1]):
                    p1, = plt.plot(x,Ytest[:test_length,i],color='blue',linewidth=1,label='GT')
                    p2, = plt.plot(x,(predict[:test_length,i]),color='red',linewidth=1,label='Predict')
                    plt.xlabel("mins/time")
                    plt.ylabel(columns[i])
                    plt.legend([p2, p1], ["Predict", "GT"], loc='upper left')
                    plt.savefig(fig_path+'/'+columns[i]+'_'+str(epoch)+'.png')
                    plt.close('all')
    return rse, corr, mae, mse, rmse

def evaluate1(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        predict = None
        test = None
        
        for X, Y in data.get_batches(X, Y, batch_size, False):
            output = model(X)
            if output.shape != Y.shape:
                break
            if predict is None:
                predict = output
                test = Y
            else:
                predict = torch.cat((predict,output))
                test = torch.cat((test, Y))
            if data.pre_length > 1:
                scale = data.scale.expand(output.size(0), data.pre_length,data.m)
            else:
                scale = data.scale.expand(output.size(0), data.m)
            total_loss += evaluateL2(output * scale, Y * scale).data
            total_loss_l1 += evaluateL1(output * scale, Y * scale).data
            n_samples += (output.size(0) * data.m*data.pre_length)
        rse = math.sqrt(total_loss / n_samples)/data.rse
        rae = (total_loss_l1/n_samples)/data.rae
        
        predict = predict.data.cpu().numpy()
        Ytest = test.data.cpu().numpy()
        sigma_p = (predict).std(axis = 0)
        sigma_g = (Ytest).std(axis = 0)
        mean_p = predict.mean(axis = 0)
        mean_g = Ytest.mean(axis = 0)
        index = (sigma_g!=0)
        correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g)
        correlation = (correlation[index]).mean()
    return rse, rae, correlation

def train(data, X, Y, model, criterion, optim, batch_size,epoch,data_nums,graph_flag,writer):
    model.train()
    total_loss = 0
    n_samples = 0
    batch = 0
    batchs = int(data_nums // batch_size)
    for X, Y in data.get_batches(X, Y, batch_size, True):
        optim.zero_grad()
        output = model(X)
        if output.shape[0] != Y.shape[0]:
            break

        # if graph_flag:
        #     writer.add_graph(model,X)
        #     graph_flag = False
        #     print('fuck')
        if data.pre_length > 1:
            scale = data.scale.expand(output.size(0), data.pre_length,data.m)
        else:
            scale = data.scale.expand(output.size(0), data.m)
        loss = criterion(output * scale, Y * scale)
        loss.backward()
        optim.step()
        total_loss += loss
        # n_samples += (output.size(0) * data.m*data.pre_length)
        n_samples += 1

        print('|now epoch is {:3d} | batch is {:5d}th / {:5d}| loss is {:8.4f}'.format(epoch,batch,batchs,loss))
        batch += 1
    return total_loss / n_samples
feature_num = 4 #需要在模型中改变
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
# parser.add_argument('--data', type=str, required=True,
#                     help='location of the data file')
parser.add_argument('--model', type=str, default='myNet',
                    help='')
parser.add_argument('--hidCNN', type=int, default=32*feature_num,
                    help='number of CNN hidden units')
parser.add_argument('--feature_num', type=int, default=feature_num,)
parser.add_argument('--hidRNN', type=int, default=32*feature_num,
                    help='number of RNN hidden 128')
parser.add_argument('--window', type=int, default=96,
                    help='window size')
parser.add_argument('--CNN_kernel', type=int, default=3,
                    help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=10*60,
                    help='The window size of the highway component')
parser.add_argument('--d_model', type=int, default=32*feature_num,)
parser.add_argument('--nhead', type=int, default=8,)
parser.add_argument('--num_layers', type=int, default=3,)
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--steps',type=int,default=1,help='')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--sampleinver', type=int, default=1,)
parser.add_argument('--poolTimes', type=int, default=1,)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--L1Loss', type=bool, default=False)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default=None)
args = parser.parse_args()
args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
# set_names = ['exchange_rate','solar_AL','ETTh1','ETTh2','ETTm1','ETTm2','AIOps']
# set_names = ['ETTh1','ETTh2','ETTm1','ETTm2']
set_names = ['ETTh1']
for set_name in set_names:
    if not os.path.exists(set_name):
        os.mkdir(set_name)
    train_fold = set_name + '/' +time.strftime("%Y-%m-%d %X", time.localtime())
    os.mkdir('./'+ train_fold)
    os.mkdir('./'+train_fold+'/save')
    os.mkdir('./'+train_fold+'/figs')
    data = np.load('./data/'+set_name+'.npy')
    Data = Data_utility(data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.steps, args.normalize)
    print(Data.rse)
    model = eval(args.model).Model(args, Data)
    if args.cuda:
        model.cuda()
    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)
    if args.L1Loss:
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    evaluateL2 = nn.MSELoss()
    evaluateL1 = nn.L1Loss()
    if args.cuda:
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()
        
        
    best_val = 10000000
    # optim = Optim.Optim(
    #     model.parameters(), args.optim, args.lr, args.clip, lr_decay=0.9, start_decay_at=2
    # )
    optimer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimer, step_size=6, gamma=0.1)
    # At any point you can hit Ctrl + C to break out of training early.
    print('begin training')
    # log_txt = open("./"+train_fold+"/log.txt","a")
    agrs_txt = open("./"+train_fold+"/agrs.txt","w")
    agrs_txt.write(str(args))
    agrs_txt.close()
    graph_flag = True
    writer = SummaryWriter("./"+train_fold+'/logs')
    for epoch in range(1, args.epochs+1):
        with open("./"+train_fold+"/log.txt","a") as f:
            epoch_start_time = time.time()
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optimer, args.batch_size, epoch, Data.train[0].shape[0],graph_flag= graph_flag ,writer = writer)
            # end = time.time()
            # time_train = end-epoch_start_time
            # val_loss, val_rae, val_corr = evaluate1(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args.batch_size)
            # print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.8f} | valid rse {:5.8f} | valid rae {:5.8f} | valid corr  {:5.8f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
            # log_txt.write('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.8f} | valid rse {:5.8f} | valid rae {:5.8f} | valid corr  {:5.8f} \n'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
            rse, corr, mae, mse, rmse  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size,train_fold+'/figs/',1000,epoch)
            # time_inferce = time.time()-end
            # resource.show_info('./',time_train, time_inferce, nParams)
            # while True:
            #     pass 
            print('rse:{},corr:{},mae:{},mse:{},rmse:{}.'.format(rse,corr,mae,mse,rmse))
            f.write('rse:{},corr:{},mae:{},mse:{},rmse:{}.\n'.format(rse,corr,mae,mse,rmse))

            writer.add_scalar('train_loss', train_loss, epoch)
            # writer.add_scalar('val_loss', val_loss, epoch)

            # writer.add_scalar('val_rae', val_rae, epoch)
            # writer.add_scalar('val_corr', val_corr, epoch)

            # writer.add_scalar('test_acc', test_acc, epoch)
            # writer.add_scalar('test_rae', test_rae, epoch)
            # writer.add_scalar('test_corr', test_corr, epoch)

            # Save the model if the validation loss is the best we've seen so far.

            # if val_loss < best_val:
            #     with open(args.save, 'wb') as f:
            #         torch.save(model, f)
            #     best_val = val_loss
            if epoch % 1 == 0:
                with open('./'+train_fold+"/save/"+str(epoch)+".pth",'wb') as f:
                    torch.save(model.state_dict(),f)
            scheduler.step()

# Load the best saved model.

# test_acc, test_rae, test_corr  = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1, args.batch_size)
# print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
# log_txt.write("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}\n".format(test_acc, test_rae, test_corr))
