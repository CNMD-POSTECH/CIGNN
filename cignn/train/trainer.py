import os
import time
import torch
import shutil
import numpy as np
import pandas as pd
import contextlib2 as contextlib
from torch_ema import ExponentialMovingAverage

class ModelTrainer:
    def __init__(self, path, model, target_list, criterion,device,use_ema=True,ema_decay=0.99, optimizer=None, rmse_loss=False, normalizers=None, print_freq=10):
        self.path=path
        self.model=model
        self.device=device
        self.target_list=target_list 
        self.criteria=criterion
        self.optimizer=optimizer
        self.rmse_loss=rmse_loss
        self.normalizers=normalizers
        self.print_freq=print_freq
        self.use_ema =use_ema
        self.ema_decay=ema_decay
        self.target_processors={
            'force': self.process_force,
            'energy': self.process_energy,
            'charge': self.process_charge,
            'chi_std':self.process_chi_std
        }
        self.ema=None
        if use_ema == True and self.ema == None:        
            self.ema=ExponentialMovingAverage(self.model.parameters(),
                            decay=self.ema_decay,
                            use_num_updates=True)
            
        self.normalizers['energy']=self.normalizers['force']
        
    def reset(self):
        self.losses={k: AverageMeter() for k in self.target_list}
        self.maes={k: AverageMeter() for k in self.target_list}
        self.losses['total']= AverageMeter()
        self.maes['total']= AverageMeter()
        self.data_time=AverageMeter()
        self.batch_time=AverageMeter()
     
    def process_force(self, target_key, target_data, output, save_output):
        target=target_data[target_key]
        normed_target=self.normalizers[target_key].scale(target)
        norm_output=output[target_key]
        denorm_output=self.normalizers[target_key].descale(norm_output)
        return self.process_common(target_key, normed_target, norm_output, denorm_output,target, save_output)

    def process_energy(self, target_key, target_data, output, save_output):
        atom_number=torch.unique(output['crystal_atom_idx'], return_counts=True)[1].reshape(-1, 1)
        target=target_data[target_key] / atom_number #per atom energy
        normed_target=self.normalizers['force'].scale(target)
        target= target*atom_number # total_energy

        norm_output=output[target_key] # per atom energy
        denorm_output=self.normalizers['force'].descale(norm_output)
        denorm_output=denorm_output*atom_number # total_energy
        return self.process_common(target_key, normed_target, norm_output, denorm_output,target, save_output)

    def process_charge(self, target_key, target_data, output, save_output):
        target=target_data[target_key]
        normed_target=target
        norm_output=output[target_key]
        denorm_output=norm_output
        return self.process_common(target_key, normed_target, norm_output, denorm_output,target, save_output)
    
    def process_chi_std(self, target_key, target_data, output, save_output):
        target=target_data[target_key]
        normed_target=target
        norm_output=output[target_key]
        denorm_output=norm_output
        return self.process_common(target_key, normed_target, norm_output, denorm_output,target, save_output)

    def process_common(self, target_key, normed_target, norm_output, denorm_output,target, save_output):
        loss=self.criteria[target_key](norm_output.to(torch.double), normed_target.to(torch.double))
        if self.rmse_loss:
            loss=torch.sqrt(loss)
        self.losses[target_key].update(loss.data.cpu(), normed_target.size(0))
        mae_error=mae(denorm_output.data.cpu(), target.cpu())
        self.maes[target_key].update(mae_error, target.size(0))
        if save_output:
            self.save_output(target_key, target, denorm_output)
        return (loss ,mae_error)

    def save_output(self, target_key, target, denorm_output):
        if target.dim() == 1:
            target_data_np=target.detach().cpu().numpy().reshape(-1,1)
            denorm_output_np=denorm_output.data.detach().cpu().numpy().reshape(-1,1)
            columns_label=[f'{i}_{j}' for i in ['pred','targ'] for j in range(target_data_np.shape[-1])]
        else:
            target_data_np=target.detach().cpu().numpy()
            denorm_output_np=denorm_output.data.detach().cpu().numpy()
            columns_label=[f'{i}_{j}' for i in ['pred','targ'] for j in range(target_data_np.shape[-1])]
        df=pd.DataFrame(np.concatenate([denorm_output_np, target_data_np], axis=1), columns=columns_label)
        df.to_csv(f'./{self.path}/{target_key}_results.csv')

    def run(self, loader, is_train=False, save_output=False, epoch=None, run_type=None):
        self.model.train() if is_train else self.model.eval()
        if is_train:
            cm=contextlib.nullcontext()
        else:
            cm=self.ema.average_parameters()
        end=time.time()
        with cm:
            self.reset()
            for i, data in enumerate(loader):
                self.optimizer.zero_grad(set_to_none=True) #TODO test 필요
                self.data_time.update(time.time() - end)
                for key, value in data.items():
                    data[key]=value.to(self.device)
                batch_number=data['energy'].shape[0]
                target_data={k: data.pop(k) if k in data else torch.zeros(batch_number,device=self.device) for k in self.target_list}

                output=self.model(data)

                total_loss=0
                total_mae =0
                for target in self.target_list:
                    loss , mae_error=self.target_processors[target](target, target_data, output, save_output)
                    if loss > 1000:
                        print(f'loss {loss} is too large, structure check {output["structure_idx"]}')
                        for key, value in data.items():
                            data[key]=value.cpu()
                        np.save('./error.npy',data)
                    total_loss=total_loss + loss * self.loss_weights[target]
                    total_mae=total_mae + mae_error
                self.losses['total'].update(total_loss,len(self.target_list))
                self.maes['total'].update(total_mae,len(self.target_list))

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000)
                    self.optimizer.step()
                    self.ema.update()

                self.batch_time.update(time.time() - end)
                end=time.time()
                if i % self.print_freq == 0:
                # if i % 1 == 0:
                    self.print_progress(epoch, i, len(loader), run_type)
                    # Explicitly delete large variables to free memory

        return self.batch_time.avg, self.losses, self.maes

    def print_progress(self, epoch, i, loader_len, run_type):
        print(
            f'Epoch [{run_type}] : [{epoch}][{i}/{loader_len}]\t'
            f'Time {self.batch_time.val:.3f} ({self.batch_time.avg:.3f})\t'
            f'Data {self.data_time.val:.3f} ({self.data_time.avg:.3f})\t'
            + ''.join([f'{target.capitalize()} Loss {self.losses[target].val:.4f} ({self.losses[target].avg:.4f})\t'
                       for target in self.losses.keys()])
            + ''.join([f'{target.capitalize()} MAE {self.maes[target].val:.4f} ({self.maes[target].avg:.4f})\t'
                       for target in self.maes.keys()])
        )

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar', bestname='model_best.pth.tar'):
        filename=os.path.join(self.path, filename)
        if self.use_ema == True:
            cm=self.ema.average_parameters()
        else:
            cm=contextlib.nullcontext()
        with cm:
            torch.save(state, filename)
            if is_best:
                shutil.copyfile(filename, bestname)

    def train(self, loader, epoch, **kwargs):
        self.loss_weights={k: kwargs.get(f'loss_weight_{k}') for k in self.target_list}

        return self.run(loader=loader, is_train=True, epoch=epoch, run_type='train')

    def validate(self, loader, direct, epoch, **kwargs):
        self.loss_weights={k: kwargs.get(f'loss_weight_{k}') for k in self.target_list}
        if direct:
            with torch.no_grad():  # Disable gradient computation
                return self.run(loader=loader, is_train=False, epoch=epoch, run_type='valid')
        else:
            return self.run(loader=loader, is_train=False, epoch=epoch, run_type='valid')
           
    def test(self, loader, direct, epoch, **kwargs):
        self.loss_weights={k: kwargs.get(f'loss_weight_{k}') for k in self.target_list}
        if direct:
            with torch.no_grad():  # Disable gradient computation
                return self.run(loader=loader, is_train=False, epoch=epoch,save_output=True, run_type='test')
        else:     
            return self.run(loader=loader, is_train=False, epoch=epoch,save_output=True,  run_type='test')
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self, val, n=1):
        self.val=val
        self.sum += val * n
        self.count += n
        self.avg=self.sum / self.count

def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target
    Parameters
    ----------
    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))