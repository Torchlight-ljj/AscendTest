import argparse
import math
import time

import torch
import torch.nn as nn
from models import FC,myNet
import numpy as np
import importlib
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from utils import *
import Optim
import torch.nn.functional as F
from metrics import *


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
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--poolTimes', type=int, default=1,)
parser.add_argument('--sampleinver', type=int, default=1,)
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--npu', type=str, default=True)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--L1Loss', type=bool, default=False)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default=None)
args = parser.parse_args()
def pred_plot(data, X, Y, model, batch_size, figs_fold, logs_file,preds_list,set_name):
    model.eval()
    with torch.no_grad():
        for preds in preds_list:
            output = None
            start = None

            output_all = []
            true_all = []

            true = None
            fuck = None
            num = 0
            steps = data.pre_length
            for X_data, Y_data in data.get_batches(X, Y, batch_size, False):
                if num ==0 :
                    time_begin = time.time()

                    start = X_data
                    output = model(X_data)
                    output_all.append(output)
                    output = output.unsqueeze(0)
                    time_mid = time.time()
                true_all.append(Y_data)
                num += 1
                if num == (preds+1)*steps:
                    break
                # print(num)
            time_end1 = time.time()

            for i in range(preds):
                mid = start[:,steps:,:]
                start = torch.cat([mid,output], dim = 1)
                # print(X.shape)
                # output = model(output[:,wins:,:])
                output = model(start)
                output_all.append(output)
                output = output.unsqueeze(0)
                # print(i)
            time_end = time.time()
            print('total_time:',time_mid-time_begin+time_end-time_end1)
            for i in range(len(true_all)):
                if i == 0:
                    true = true_all[i]
                else:
                    true = torch.cat([true,true_all[i]],dim=0)

            for i in range(len(output_all)):
                if i == 0:
                    fuck = output_all[i]
                    # print(fuck.shape)
                else:
                    fuck = torch.cat([fuck,output_all[i]],dim=0)
            # fuck = fuck.unsqueeze(1)
            fuck = fuck.data.cpu().numpy()
            true = true.data.cpu().numpy()
            # print(fuck.shape,true.shape)
            rse, corr, mae, mse, rmse, mape, mspe = metric(fuck, true)
            print('pred_steps:{},rse:{},corr:{},mae:{},mse:{},rmse:{}.'.format(preds,rse,corr,mae,mse,rmse))
            logs_file.write('pred_steps:{},rse:{},corr:{},mae:{},mse:{},rmse:{}.\n'.format(preds,rse,corr,mae,mse,rmse))
        
            plt_fold = figs_fold+str(preds)+'/'
            if not os.path.exists(plt_fold):
                os.mkdir(plt_fold)
            x = np.linspace(0,fuck.shape[0],fuck.shape[0])  #设置横轴的取值点
            # columns =['SP1A_DASD_RESP','SP1A_DASD_RATE','SP1B_DASD_RESP','SP1B_DASD_RATE','SP1C_DASD_RESP',
            #     'SP1C_DASD_RATE','SP1D_DASD_RESP','SP1D_DASD_RATE','SP1A_MEM','SP1B_MEM','SP1C_MEM','SP1D_MEM','N_TASKS','TPS','SP1A_THOUT','SP1B_THOUT','SP1C_THOUT','SP1D_THOUT','SYSPLEX_MIPS','RESP_TIME']
            # columns = ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
            columns = data.columns
            for i in range(fuck.shape[1]):
                p2, = plt.plot(x,(fuck[:,i]),color='red',linewidth=1,label='Predict')
                p1, = plt.plot(x,(true[:,i]),color='blue',linewidth=1,label='GT')
                plt.xlabel("mins/time")
                plt.ylabel(columns[i])
                plt.legend([p2,p1], ["Predict","GT"], loc='upper left')
                plt.savefig(plt_fold+columns[i]+'_'+'.png')
                plt.close('all')
            if preds == 288:
                np.savetxt(os.path.join(set_name,'continue_trues.csv'),true,delimiter=',')
                np.savetxt(os.path.join(set_name,'continue_preds.csv'),fuck,delimiter=',')

def evaluate(data, X, Y, model, batch_size, figs_fold = None, test_length = None, epoch = None):
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
        predict = predict.data.cpu().numpy()[1:test_length]
        Ytest = test.data.cpu().numpy()[1:test_length]
        np.savetxt(os.path.join(figs_fold,'compare_trues.csv'),Ytest,delimiter=',')
        np.savetxt(os.path.join(figs_fold,'compare_preds.csv'),predict,delimiter=',')


def pred_plot_ensemble(data, X, Y, model, evaluateL2, evaluateL1, batch_size, figs_fold, logs_file,preds_list,skip):
    model.eval()
    with torch.no_grad():
        for preds in preds_list:
            output = None
            start = None
            in_skip = skip

            output_all = []
            true_all = []

            true = None
            fuck = None
            num = 0
            steps = data.pre_length
            for X_data, Y_data in data.get_batches(X, Y, batch_size, False):
                in_skip -= 1
                if in_skip > 0:
                    continue
                if num ==0 :
                    time_begin = time.time()
                    start = X_data
                    for i in range(1,preds+2):
                        model.load_state_dict(torch.load("./ensemble/"+str(i)+'/save/5.pth'))
                        output = model(X_data)
                        output_all.append(output)
                        # output = output.unsqueeze(0)
                    time_mid = time.time()
                true_all.append(Y_data)
                num += 1
                if num == (preds+1)*steps:
                    break
                # print(num)
            time_end1 = time.time()

            # for i in range(preds):
            #     mid = start[:,steps:,:]
            #     start = torch.cat([mid,output], dim = 1)
            #     # print(X.shape)
            #     # output = model(output[:,wins:,:])
            #     output = model(start)
            #     output_all.append(output)
            #     output = output.unsqueeze(0)
                # print(i)
            time_end = time.time()
            print('total_time:',time_mid-time_begin+time_end-time_end1)
            for i in range(len(true_all)):
                if i == 0:
                    true = true_all[i]
                else:
                    true = torch.cat([true,true_all[i]],dim=0)

            for i in range(len(output_all)):
                if i == 0:
                    fuck = output_all[i]
                    # print(fuck.shape)
                else:
                    fuck = torch.cat([fuck,output_all[i]],dim=0)
            # fuck = fuck.unsqueeze(1)
            fuck = fuck.data.cpu().numpy()
            true = true.data.cpu().numpy()
            # print(fuck.shape,true.shape)
            rse, corr, mae, mse, rmse, mape, mspe = metric(fuck, true)
            print('pred_steps:{},rse:{},corr:{},mae:{},mse:{},rmse:{}.'.format(preds,rse,corr,mae,mse,rmse))
            logs_file.write('pred_steps:{},rse:{},corr:{},mae:{},mse:{},rmse:{}.\n'.format(preds,rse,corr,mae,mse,rmse))
        
            plt_fold = figs_fold+str(preds)+'/'
            if not os.path.exists(plt_fold):
                os.mkdir(plt_fold)
            x = np.linspace(0,fuck.shape[0],fuck.shape[0])  #设置横轴的取值点
            columns =['SP1A_DASD_RESP','SP1A_DASD_RATE','SP1B_DASD_RESP','SP1B_DASD_RATE','SP1C_DASD_RESP',
                'SP1C_DASD_RATE','SP1D_DASD_RESP','SP1D_DASD_RATE','SP1A_MEM','SP1B_MEM','SP1C_MEM','SP1D_MEM','N_TASKS','TPS','SP1A_THOUT','SP1B_THOUT','SP1C_THOUT','SP1D_THOUT','SYSPLEX_MIPS','RESP_TIME']
            # columns = ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
            for i in range(fuck.shape[1]):
                p2, = plt.plot(x,(fuck[:,i]),color='red',linewidth=1,label='Predict')
                p1, = plt.plot(x,(true[:,i]),color='blue',linewidth=1,label='GT')
                plt.xlabel("mins/time")
                plt.ylabel(columns[i])
                plt.legend([p2,p1], ["Predict","GT"], loc='upper left')
                plt.savefig(plt_fold+columns[i]+'_'+'.png')
                plt.close('all')
        print('finished:',preds)

set_names = ['exchange_rate','solar_AL','ETTh1','ETTh2','ETTm1','ETTm2','AIOps'] ##solar_AL:max=16
weights = {"AIOps":"/save/3.pth","ETTh1":"/save/12.pth","ETTh2":"/save/15.pth","ETTm1":"/save/14.pth","ETTm2":"/save/13.pth","exchange_rate":"/save/7.pth","solar_AL":"/save/8.pth"}
times = {"AIOps":"/2022-05-11 13:15:40","ETTh1":"/2022-05-14 14:50:19","ETTh2":"/2022-05-11 00:40:10","ETTm1":"/2022-05-11 00:53:45",
"ETTm2":"/2022-05-11 01:47:33","exchange_rate":"/2022-05-10 23:28:27","solar_AL":"/2022-05-10 10:26:01"}
set_name = 'ETTh1'
data = np.load('../MTS_data/'+set_name+'.npy')
Data = Data_utility(data, 0.6, 0.2, args.npu, args.horizon, args.window, args.steps, args.normalize); #SPLITS THE DATA IN TRAIN AND VALIDATION SET, ALONG WITH OTHER THINGS, SEE CODE FOR MORE
print(Data.dat.shape);

model = eval(args.model).Model(args, Data)
model_file = set_name +times[set_name]
figs_fold = model_file+"/test_figs/"
# skip = 500
# figs_fold = "./ensemble/figs" + str(skip) +'/'
# model.load(model_file+"/save/1.pt")
# model = torch.load(model_file+"/save/1.pt")
import os
if not os.path.exists(figs_fold):
    os.mkdir(figs_fold)
logs_file = open(figs_fold+'/log.txt','a')

if args.npu:
    model.npu()
model.load_state_dict(torch.load(model_file+weights[set_name]))
##single-step output
evaluate(Data, Data.test[0], Data.test[1], model,  args.batch_size, set_name,1001)

###iterative output
preds_list = [12,48,96,144,192,240,288]
pred_plot(Data, Data.valid[0], Data.valid[1], model, 1, figs_fold, logs_file, preds_list,set_name)