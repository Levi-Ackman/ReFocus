from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate,visual_forecast,visual_fea,plot_heatmap
from utils.metrics import metric,MSE,MAE
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
import torch.nn.functional as F

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        print("Model total parameters: {:.2f} M".format(sum(p.numel() for p in model.parameters())/1e+6))
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_loader = data_provider(self.args, flag)
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss() if (self.args.loss=='MSE' or self.args.loss=='mse') else nn.L1Loss()
        return criterion

    def vali(self, vali_loader):
        self.model.eval()
        with torch.no_grad():
            preds = []
            trues = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
        if len(preds)>0:
            preds=np.concatenate(preds, axis=0)
            trues=np.concatenate(trues, axis=0)
        else:
            preds=preds[0]
            trues=trues[0]
        mse = MSE(preds, trues) 
        mae=MAE(preds, trues)
        self.model.train()
        return mse,mae

    def train(self, setting):
        self.freq_augmentation=1
        self.sigma=1
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='val')
        test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        epoch_count=0
        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, :].float().to(self.device)
                n_batch, n_vars = batch_y.shape[0], batch_y.shape[2]
                
                if self.freq_augmentation and epoch_count%2==1:
                    perm = torch.randperm(n_vars, device=self.device)
                    pseudo_channel = torch.normal(mean=0, std=self.sigma, size=(n_batch, n_vars), device=self.device).unsqueeze(-2)
                    batch_x = batch_x + batch_x[:, :, perm] * pseudo_channel
                    batch_y = batch_y + batch_y[:, :, perm] * pseudo_channel

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_mse,vali_mae = self.vali( vali_loader)
            test_mse,test_mae = self.vali( test_loader)

            print("Epoch: {}, Steps: {} | Train Loss: {:.4f} Val Loss: {:.3f} Test Loss: {:.3f}".format(epoch + 1, train_steps, train_loss, vali_mse, test_mse))
            early_stopping(vali_mse, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            epoch_count += 1
        torch.cuda.empty_cache()
        ## Uncomment below code for save space on device
        # if os.path.exists(os.path.join(os.path.join(path, 'checkpoint.pth'))):
        #     os.remove(os.path.join(os.path.join(path, 'checkpoint.pth')))
        #     print('Model weights deleted.')

    def test(self, setting, test=1):
        test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        self.model.eval()
        with torch.no_grad():
            preds = []
            trues = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
        if len(preds)>0:
            preds=np.concatenate(preds, axis=0)
            trues=np.concatenate(trues, axis=0)
        else:
            preds=preds[0]
            trues=trues[0]
        print('test shape:', preds.shape, trues.shape)
        mae, mse = metric(preds, trues)
        print('mse:{:.3f}, mae:{:.3f}'.format(mse, mae))
        
        # Uncomment what follows to save the test dict: 
        dict_path = f'./test_dict/{self.args.data_path[:-4]}/{self.args.seq_len}_{self.args.pred_len}/'
        if not os.path.exists(dict_path):
                os.makedirs(dict_path)
        my_dict = {
            'mse': float(round(mse, 3)),
            'mae': float(round(mae, 3)),
        }
        with open(os.path.join(dict_path, 'records.json'), 'w') as f:
            json.dump(my_dict, f)
        f.close()
        
        return 
    
    def visual_weight(self, setting, test=1):
        visual_path = f'./visual/weight/{self.args.data_path[:-4]}/{self.args.seq_len}_{self.args.pred_len}/'
        if not os.path.exists(visual_path):
                os.makedirs(visual_path)
        path = os.path.join(self.args.checkpoints, setting)
        if test:
            print('loading model..............\n')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        self.model.eval()
        with torch.no_grad():
            batch_x = torch.zeros(self.args.batch_size,self.args.seq_len, self.args.enc_in).float().to(self.device)
            outputs= self.model(batch_x)
            bias=outputs[0,:,0]
            weight = torch.zeros(self.args.seq_len, self.args.pred_len).float().to(self.device)
            for i in range(self.args.seq_len):
                batch_x = torch.zeros(self.args.batch_size,self.args.seq_len, self.args.enc_in).float().to(self.device)
                batch_x[:,i, :] = 1
                outputs= self.model(batch_x)
                weight[i,:]=outputs[0,:,0]-bias
                
            plot_heatmap(weight.detach().cpu().numpy(),visual_path+'w.png',figsize=(6.4+4.8*(self.args.pred_len//96-1), 4.8))
            
            mean = weight.mean(dim=1, keepdim=True)  
            std = weight.std(dim=1, keepdim=True)    
            weight = (weight - mean) / (std + 1e-10) 
            
            u,singularvalue,v = torch.svd(weight)
            sum_singular = torch.max(singularvalue.real)
            print("Sum of singularvalue of the weight matrix:", sum_singular)
            
            weight_normalized = F.softmax(weight,1)
            entropy_per_row = -torch.sum(weight_normalized * torch.log(weight_normalized + 1e-10), dim=1)  
            mean_entropy = torch.mean(entropy_per_row)
            print("entropy of the weight matrix :", mean_entropy)

            return 
        
    def visual_feature(self, setting, test=1):
        test_loader = self._get_data(flag='test')
        visual_path = f'./visual/feature/{self.args.data_path[:-4]}/{self.args.seq_len}_{self.args.pred_len}/'
        if not os.path.exists(visual_path):
                os.makedirs(visual_path)
        path = os.path.join(self.args.checkpoints, setting)
        if test:
            print('loading model..............\n')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        self.model.eval()
        with torch.no_grad():
            channel=[2,3,4]
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                x_enc=batch_x[1,:,:].unsqueeze(0)
                step=4
                if i % step == 0:
                    x_emb=self.model.embed(x_enc.transpose(1,2))
                    _, feature= self.model(x_enc,1)
                    x_enc=x_enc.squeeze(0).detach().cpu().numpy()
                    feature=feature.squeeze(0).detach().cpu().numpy()
                    x_emb=x_emb.squeeze(0).detach().cpu().numpy()
                    
                    visual_fea(x_enc,channel,name=os.path.join(visual_path, str(i)+'_input.png'))
                    visual_fea(x_emb,channel,name=os.path.join(visual_path, str(i)+'_CI_feature.png'))
                    visual_fea(feature,channel,name=os.path.join(visual_path, str(i)+'_EKPB_feature.png'))
                    
                    for j in range(len(channel)):
                        visual_forecast([x_enc[:,channel[j]],f'Variate{j+1}',1,],name=os.path.join(visual_path, str(i)+'_'+f'Variate {j+1}.png'))
                        
            return 
        
    def visual_forecasting(self, setting, test=1):
        visual_path = f'./visual/forecasting/{self.args.data_path[:-4]}/{self.args.seq_len}_{self.args.pred_len}/'
        if not os.path.exists(visual_path):
                os.makedirs(visual_path)
        test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if test:
            print('loading model..............\n')
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                
                outputs = self.model(batch_x)
                step=4
                if i % step == 0:
                    pred = outputs.detach().cpu().numpy()[0, :, -1]
                    true = batch_y.detach().cpu().numpy()[0, -self.args.pred_len:, -1]
                    input = batch_x.detach().cpu().numpy()[0, :, -1]
                    pd = np.concatenate([input, pred])
                    tg= np.concatenate([input, true])

                    visual_forecast([pd,'forecast',1],[tg,'target',4], \
                               name=os.path.join(visual_path, str(i) + '_forecast.png'))
        return 