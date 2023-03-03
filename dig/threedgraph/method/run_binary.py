import time
import os
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
import numpy as np
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score


class run_binary():
    r"""
    The base script for running different 3DGN methods.
    """
    def __init__(self):
        pass
    
    #def binary_auc(y_pred, y_test):
    #    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    #    auc = roc_auc_score(y_test, y_pred_tag)
    #    return auc
    
    def run_binary(self, device, train_loader, val_loader, test_loader, model, loss_func, evaluation, epochs=500, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=0, 
        energy_and_force=False, p=100, save_dir='', log_dir=''):
        r"""
        The run script for training and validation.
        
        Args:
            device (torch.device): Device for computation.
            train_dataset: Training data.
            valid_dataset: Validation data.
            test_dataset: Test data.
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            loss_func (function): The used loss funtion for training.
            evaluation (function): The evaluation function. 
            epochs (int, optinal): Number of total training epochs. (default: :obj:`500`)
            batch_size (int, optinal): Number of samples in each minibatch in training. (default: :obj:`32`)
            vt_batch_size (int, optinal): Number of samples in each minibatch in validation/testing. (default: :obj:`32`)
            lr (float, optinal): Initial learning rate. (default: :obj:`0.0005`)
            lr_decay_factor (float, optinal): Learning rate decay factor. (default: :obj:`0.5`)
            lr_decay_step_size (int, optinal): epochs at which lr_initial <- lr_initial * lr_decay_factor. (default: :obj:`50`)
            weight_decay (float, optinal): weight decay factor at the regularization term. (default: :obj:`0`)
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            save_dir (str, optinal): The path to save trained models. If set to :obj:`''`, will not save the model. (default: :obj:`''`)
            log_dir (str, optinal): The path to save log files. If set to :obj:`''`, will not save the log files. (default: :obj:`''`)
        
        """        
        model = model.to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f'#Params: {num_params}')
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

        #train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        #valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
        #test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
        best_valid = float('inf')
        best_test = float('inf')
            
        if save_dir != '':
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        if log_dir != '':
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)
        
        train_scores, train_losses, val_scores, val_losses, test_scores, test_losses = [], [], [], [], [], []
        for epoch in range(1, epochs + 1):
            print("\n=====Epoch {}".format(epoch), flush=True)
            
            print('\nTraining...', flush=True)
            train_auc, train_loss = self.train(model, optimizer, train_loader, energy_and_force, p, loss_func, device)
            train_scores.append(train_auc)
            train_losses.append(train_loss)
            print('\n\nEvaluating...', flush=True)
            valid_auc ,valid_loss = self.val(model, val_loader, energy_and_force, p, evaluation, device)
            val_scores.append(valid_auc)
            val_losses.append(valid_loss)
            print('\n\nTesting...', flush=True)
            test_auc, test_loss = self.val(model, test_loader, energy_and_force, p, evaluation, device)
            test_scores.append(test_auc)
            test_losses.append(test_loss)
            print()
            print({'Train Loss': train_loss, 'Train AUC': train_auc, 
                   'Validation Loss': valid_loss, 'Validation AUC': valid_auc, 
                   'Test Loss': test_loss, 'Test AUC': test_auc})

            if log_dir != '':
                writer.add_scalar('train_loss', train_loss, epoch)
                writer.add_scalar('valid_loss', valid_loss, epoch)
                writer.add_scalar('test_loss', test_loss, epoch)
            
            if valid_loss < best_valid:
                best_valid_loss = valid_loss
                best_valid_auc = valid_auc
                best_test_loss = test_loss
                best_test_auc = test_auc
                
                if save_dir != '':
                    print('Saving checkpoint...')
                    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_valid_loss': best_valid_loss, 'best_valid_auc': best_valid_auc, 'num_params': num_params}
                    torch.save(checkpoint, os.path.join(save_dir, 'valid_checkpoint.pt'))

            scheduler.step()

        print(f'Best validation Loss so far: {best_valid_loss}')
        print(f'Best validation AUC so far: {best_valid_auc}')
        print(f'Test Loss when got best validation result: {best_test_loss}')
        print(f'Test AUC when got best validation result: {best_test_auc}')
        if log_dir != '':
            writer.close()
        return train_scores, train_losses, val_scores, val_losses, test_scores, test_losses

    def train(self, model, optimizer, train_loader, energy_and_force, p, loss_func, device):
        r"""
        The script for training.
        
        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            optimizer (Optimizer): Pytorch optimizer for trainable parameters in training.
            train_loader (Dataloader): Dataloader for training.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy during training. (default: :obj:`100`)
            loss_func (function): The used loss funtion for training. 
            device (torch.device): The device where the model is deployed.

        :rtype: Traning loss. ( :obj:`mae`)
        
        """   
        #print(loss_func)
        model.train()
        loss_accum = 0
        auc_accum = 0
        for step, batch_data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            out = model(batch_data)
            print('batch_data', batch_data.y)
            print('batch_data shape', batch_data.y.shape)
            print('out', out)
            print('out shape', out.shape)
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out),create_graph=True,retain_graph=True)[0]
                e_loss = loss_func(out, batch_data.y.unsqueeze(1))
                f_loss = loss_func(force, batch_data.force)
                loss = e_loss + p * f_loss
            else:
                loss = loss_func(out, batch_data.y.unsqueeze(1))  # binary
                #print(out)
                # AUC
                #print(out.shape)
                #print(batch_data.y.unsqueeze(1).shape)
                auc = roc_auc_score(batch_data.y.unsqueeze(1).cpu().detach().numpy(), out.cpu().detach().numpy())
                #print(auc)
                print('train', batch_data.y.view(-1))
            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()
            auc_accum += auc
            print('train', step)
        return auc_accum / (step + 1), loss_accum / (step + 1)

    def val(self, model, data_loader, energy_and_force, p, evaluation, device):
        r"""
        The script for validation/test.
        
        Args:
            model: Which 3DGN model to use. Should be one of the SchNet, DimeNetPP, and SphereNet.
            data_loader (Dataloader): Dataloader for validation or test.
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the minus derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)    
            p (int, optinal): The forces’ weight for a joint loss of forces and conserved energy. (default: :obj:`100`)
            evaluation (function): The used funtion for evaluation.
            device (torch.device, optional): The device where the model is deployed.

        :rtype: Evaluation result. ( :obj:`mae`)
        
        """   
        model.eval()

        preds = torch.Tensor([]).to(device)
        targets = torch.Tensor([]).to(device)

        if energy_and_force:
            preds_force = torch.Tensor([]).to(device)
            targets_force = torch.Tensor([]).to(device)
        
        for step, batch_data in enumerate(tqdm(data_loader)):
            batch_data = batch_data.to(device)
            out = model(batch_data)
            if energy_and_force:
                force = -grad(outputs=out, inputs=batch_data.pos, grad_outputs=torch.ones_like(out),create_graph=True,retain_graph=True)[0]
                preds_force = torch.cat([preds_force,force.detach_()], dim=0)
                targets_force = torch.cat([targets_force,batch_data.force], dim=0)
            preds = torch.cat([preds, out.detach_()], dim=0)
            targets = torch.cat([targets, batch_data.y.unsqueeze(1)], dim=0)
            print('val', step)
            print('val', targets.view(-1))
        input_dict = {"y_true": targets, "y_pred": preds}

        #return evaluation.eval(input_dict)['mae']
        return evaluation.eval(input_dict)['auc'], evaluation.eval(input_dict)['loss']