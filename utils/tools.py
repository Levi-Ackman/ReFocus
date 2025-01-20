import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch,args):
    # lr = learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 1e-7
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 50 else args.learning_rate*0.1}  
    elif args.lradj == '7':
        lr_adjust = {epoch: args.learning_rate if epoch < 150 else args.learning_rate*0.1} 
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def visual_fea(inp,channel=[2,3,4],name='laoda.png'):
    color_dict={1:(255/255, 0/255, 0/255),
                    2:(150/255, 147/255, 200/255),
                    3:(208/255, 131/255, 131/255),
                    4:(119/255, 172/255, 190/255),
                    5:(252/255, 192/255, 212/255),
                    6:(108/255, 114/255, 195/255)}

    scaler = StandardScaler(0,1)
    inp_scaled = scaler.transform(inp)

    pca = PCA(n_components=12,svd_solver='randomized')
    inp_pca = pca.fit_transform(inp_scaled)

    tsne = TSNE(n_components=2, random_state=42,n_iter=250,method='barnes_hut')
    inp_tsne = tsne.fit_transform(inp_pca)

    plt.figure(figsize=(8, 6),dpi=600)
    plt.scatter(inp_tsne[:, 0], inp_tsne[:, 1], c=color_dict[4], s=100, edgecolor='k')
    plt.scatter(inp_tsne[channel[0], 0], inp_tsne[channel[0], 1], c=color_dict[1], marker='s', s=150, label='Variate 1')  # 正方形
    plt.scatter(inp_tsne[channel[1], 0], inp_tsne[channel[1], 1], c=color_dict[1], marker='^', s=200, label='Variate 2')  # 三角形
    plt.scatter(inp_tsne[channel[2], 0], inp_tsne[channel[2], 1], c=color_dict[1], marker='*', s=240, label='Variate 3')  # 五角星形
    plt.legend(fontsize=24)
    plt.xticks([])  
    plt.yticks([]) 
    plt.savefig(name,bbox_inches='tight')
    
def visual_forecast(inp1, inp2=None,inp3=None,inp4=None,inp5=None,inp6=None,name='./predcition_res.pdf'):
    color_dict={1:(255/255, 0/255, 0/255),
                2:(150/255, 147/255, 200/255),
                3:(208/255, 131/255, 131/255),
                4:(119/255, 172/255, 190/255),
                5:(252/255, 192/255, 212/255),
                6:(108/255, 114/255, 195/255)}
    """
    Results visualization
    """
    plt.figure(figsize=(24,12),dpi=600)
    if inp1 is not None:
        plt.plot(inp1[0], label=f'{inp1[1]}', linewidth=12,color=color_dict[inp1[2]],linestyle='-')
    if inp2 is not None:
        plt.plot(inp2[0], label=f'{inp2[1]}', linewidth=12,color=color_dict[inp2[2]],linestyle='-')
    if inp3 is not None:
        plt.plot(inp3[0], label=f'{inp3[1]}', linewidth=12,color=color_dict[inp3[2]],linestyle='-')
    if inp4 is not None:
        plt.plot(inp4[0], label=f'{inp4[1]}', linewidth=12,color=color_dict[inp4[2]],linestyle='-')
    if inp5 is not None:
        plt.plot(inp5[0], label=f'{inp5[1]}', linewidth=12,color=color_dict[inp5[2]],linestyle='-')
    if inp6 is not None:
        plt.plot(inp6[0], label=f'{inp6[1]}', linewidth=12,color=color_dict[inp6[2]],linestyle='-')
    
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    plt.legend(fontsize=42)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    # plt.xticks([])  # Hide X-axis ticks
    # plt.yticks([])  # Hide Y-axis ticks
    plt.savefig(name,bbox_inches='tight')

def plot_heatmap(matrix, filename, figsize=(6.4, 4.8), font_scale=1.2, annot=False, annot_kws=None):
    plt.clf()
    sns.set(font_scale=font_scale) 
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, cmap='coolwarm', linewidths=0.5, linecolor='gray', annot=annot, annot_kws=annot_kws)
    plt.savefig(filename, dpi=1200,bbox_inches='tight')