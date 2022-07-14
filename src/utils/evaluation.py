import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, mean_absolute_error, matthews_corrcoef
import sys
sys.path.append('../src/')
from utils.preprocessing import get_status

def evaluate_activation(model, loader, a, border):
    x_true = []
    s_true = []
    p_true = []
    s_hat = []
    
    model.eval()
    with torch.no_grad():
        for x, p, s in loader:
            x = x.unsqueeze(1)#.cuda()
            p = p.permute(0,2,1)[:,a,:]
            s = s.permute(0,2,1)[:,a,:]
            
            sh = model(x)
            sh = torch.sigmoid(sh[:,a,:])
            
            s_hat.append(sh.contiguous().view(-1).detach().cpu().numpy())
            
            x_true.append(x[:,:,border:-border].contiguous().view(-1).detach().cpu().numpy())
            s_true.append(s.contiguous().view(-1).detach().cpu().numpy())
            p_true.append(p.contiguous().view(-1).detach().cpu().numpy())
    x_true = np.hstack(x_true)
    s_true = np.hstack(s_true)
    p_true = np.hstack(p_true)
    s_hat = np.hstack(s_hat)

    return x_true, p_true, s_true, s_hat

def print_metrics(model,weights_path, x_test, min_off, min_on, max_power, appliance_list, y_test_app, y_test_stat, border, thr):
    scores = {}
    filename = weights_path
    model.load_state_dict(torch.load(filename))
    for a in range(len(appliance_list)):
        scores[a] = {}
        scores[a]['F1'] = []
        scores[a]['Precision'] = []
        scores[a]['Recall'] = []
        scores[a]['Accuracy'] = []
        scores[a]['MCC'] = []
        scores[a]['MAE'] = []
        scores[a]['SAE'] = []
        
        pm = y_test_app[appliance_list[a]].sum() / y_test_stat[appliance_list[a]].sum() / max_power
        x_true, p_true, s_true, s_hat = evaluate_activation(model, x_test, a, border)
        s_hat = get_status(s_hat, thr, min_off[a], min_on[a])
        p_hat = pm * s_hat
        scores[a]['F1'].append(f1_score(s_true, s_hat))
        scores[a]['Precision'].append(precision_score(s_true, s_hat))
        scores[a]['Recall'].append(recall_score(s_true, s_hat))
        scores[a]['Accuracy'].append(accuracy_score(s_true, s_hat))
        scores[a]['MCC'].append(matthews_corrcoef(s_true, s_hat))
        scores[a]['MAE'].append(mean_absolute_error(p_true, p_hat)*max_power)
        scores[a]['SAE'].append((p_hat.sum() - p_true.sum()) / p_true.sum())
        
    for i,a in enumerate(appliance_list):
        print()
        print(a)
        print('F1 score  : %.3f' %(scores[i]['F1'][0]))
        print('Precision : %.3f' %(scores[i]['Precision'][0]))
        print('Recall    : %.3f' %(scores[i]['Recall'][0]))
        print('Accuracy  : %.3f' %(scores[i]['Accuracy'][0]))
        print('MCC       : %.3f' %(scores[i]['MCC'][0]))
        print('MAE       : %.2f' %(scores[i]['MAE'][0]))
        print('SAE       : %.3f' %(scores[i]['SAE'][0]))