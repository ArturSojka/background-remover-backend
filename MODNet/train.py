from .data.load import create_train_dataloader, create_valid_dataloader, create_self_supervised_dataloader
from .src.trainer import supervised_training_iter, soc_adaptation_iter, blurer
from .src.models.modnet import MODNet
from torch import nn, optim
from torch.nn.functional import l1_loss, mse_loss
import torch
from tqdm import tqdm
import os
import json
import copy

def calculate_validation_metrics(modnet, valid_dataloader, device):
    modnet.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for image, gt_matte, trimap in valid_dataloader:
            image, gt_matte = image.to(device), gt_matte.to(device)
            
            # Forward pass
            _, _, pred_matte = modnet(image, True)
            
            # Calculate metrics
            mse = torch.mean(mse_loss(pred_matte, gt_matte))
            mae = torch.mean(l1_loss(pred_matte, gt_matte))
            
            batch_size = image.size(0)
            total_mse += mse.item() * batch_size
            total_mae += mae.item() * batch_size
            total_samples += batch_size
    
    modnet.train()
    return total_mse / total_samples, total_mae / total_samples

def save_checkpoint_supervised(optimizer, lr_scheduler, epoch, step, best_mae, training_history, save_dir):    
    checkpoint = {
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
        'best_mae': best_mae,
        'training_history': training_history
    }
    
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
    
def save_checkpoint_self_supervised(optimizer, epoch, step, training_history, save_dir):    
    checkpoint = {
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'training_history': training_history
    }
    
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))
    
def save_model(modnet, save_dir, is_best=False):
    if is_best:
        torch.save(modnet.state_dict(), os.path.join(save_dir, 'best.pth'))
    
    torch.save(modnet.state_dict(), os.path.join(save_dir, 'last.pth'))
    
def load_checkpoint_supervised(modnet, optimizer, lr_scheduler, save_dir):
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Cannot resume training. No checkpoint found at {checkpoint_path}")
    last_path = os.path.join(save_dir, 'last.pth')
    if not os.path.exists(last_path):
        raise ValueError(f"Cannot resume training. No model found at {last_path}")
    
    checkpoint = torch.load(checkpoint_path)
    modnet.load_state_dict(torch.load(last_path))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    
    print(f"Resumed from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
    return (checkpoint['epoch'], checkpoint['step'], checkpoint['best_mae'], checkpoint['training_history'])

def load_checkpoint_self_supervised(modnet, optimizer, save_dir):
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Cannot resume training. No checkpoint found at {checkpoint_path}")
    last_path = os.path.join(save_dir, 'last.pth')
    if not os.path.exists(last_path):
        raise ValueError(f"Cannot resume training. No model found at {last_path}")
    
    checkpoint = torch.load(checkpoint_path)
    modnet.load_state_dict(torch.load(last_path))
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Resumed from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
    return (checkpoint['epoch'], checkpoint['step'], checkpoint['training_history'])

def supervised_training(
        natural_path:str,
        synthetic_path:str,
        background_path:str,
        device:str,
        batch_size:int=16,
        epochs:int=40,
        lr:float=0.01,
        save_steps:int=5000,
        save_dir:str="./training/supervised",
        resume:bool=False
    ):
    os.makedirs(save_dir, exist_ok=True)
    train_dataloader = create_train_dataloader(natural_path, synthetic_path, background_path, batch_size)
    valid_dataloader = create_valid_dataloader(natural_path, synthetic_path, background_path, batch_size)
    
    device = torch.device(device)
    blurer.to(device)
    modnet = nn.DataParallel(MODNet(backbone_pretrained=False)).to(device)
    optimizer = optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)
    
    if resume:
        start_epoch, global_step, best_mae, training_history = load_checkpoint_supervised(
            modnet, optimizer, lr_scheduler, save_dir
        )
    else:
        start_epoch = 0
        global_step = 0
        best_mae = float('inf')
        training_history = {'train_losses': [], 'val_metrics': []}
    
    stop_file = os.path.join(save_dir, 'STOP_TRAINING')
    assert not os.path.exists(stop_file), "The 'STOP_TRAINING' file must be deleted before training can be resumed."
    print(f"Begining training. To interrupt, create a file {stop_file}")
    
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epochs
    pbar = tqdm(initial=global_step, total=total_steps)
    losses = {
        'semantic_loss': [],
        'detail_loss': [],
        'matte_loss': []
    }
    for epoch in range(start_epoch, epochs):
        
        steps_to_skip = 0
        if resume and epoch == start_epoch:
            steps_completed_in_epoch = global_step % steps_per_epoch
            steps_to_skip = steps_completed_in_epoch
            
        for image, gt_matte, trimap in train_dataloader:
            
            if steps_to_skip > 0:
                steps_to_skip -= 1
                continue
            
            image, gt_matte, trimap = image.to(device), gt_matte.to(device), trimap.to(device)
            semantic_loss, detail_loss, matte_loss = supervised_training_iter(modnet, optimizer, image, trimap, gt_matte)
            
            losses['semantic_loss'].append(semantic_loss.item())
            losses['detail_loss'].append(detail_loss.item())
            losses['matte_loss'].append(matte_loss.item())
            
            global_step+=1
            pbar.update()
            total_loss = (semantic_loss + detail_loss + matte_loss).item()
            if device.type == 'cpu':
                pbar.set_description(f"Loss: {total_loss:.4f}")
            else:
                allocated = torch.cuda.memory_allocated() / 1024**2
                cached = torch.cuda.memory_reserved() / 1024**2
                pbar.set_description(f"Loss: {total_loss:.4f} | GPU: {allocated:.0f}MB/{cached:.0f}MB")
            
            if global_step % save_steps == 0:
                print(f"\nEvaluating at step {global_step}...")
                
                val_mse, val_mae = calculate_validation_metrics(modnet, valid_dataloader, device)
                training_history['train_losses'].append({
                    'step': global_step,
                    'epoch': epoch,
                    'semantic_loss': sum(losses['semantic_loss']) / len(losses['semantic_loss']),
                    'detail_loss': sum(losses['detail_loss']) / len(losses['detail_loss']),
                    'matte_loss': sum(losses['matte_loss']) / len(losses['matte_loss'])
                })
                training_history['val_metrics'].append({
                    'step': global_step,
                    'epoch': epoch,
                    'mse': val_mse,
                    'mae': val_mae
                })
                losses = {
                    'semantic_loss': [],
                    'detail_loss': [],
                    'matte_loss': []
                }
                
                print(f"Val MSE: {val_mse:.6f}, Val MAE: {val_mae:.6f}")
                
                is_best = val_mae < best_mae
                if is_best:
                    best_mae = val_mae
                save_model(modnet,save_dir,is_best)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                if os.path.exists(stop_file):
                    save_checkpoint_supervised(optimizer, lr_scheduler, epoch, global_step, best_mae, training_history, save_dir)
                    pbar.close()
                    print(f"\nStop signal detected. Checkpoint saved.")
                    return
        lr_scheduler.step()
        
    pbar.close()
    
    print("\nTraining complete. Running final evaluation...")
    val_mse, val_mae = calculate_validation_metrics(modnet, valid_dataloader, device)
    training_history['train_losses'].append({
        'step': global_step,
        'epoch': epoch,
        'semantic_loss': sum(losses['semantic_loss']) / len(losses['semantic_loss']),
        'detail_loss': sum(losses['detail_loss']) / len(losses['detail_loss']),
        'matte_loss': sum(losses['matte_loss']) / len(losses['matte_loss'])
    })
    training_history['val_metrics'].append({
        'step': global_step,
        'epoch': epoch,
        'mse': val_mse,
        'mae': val_mae
    })
    
    print(f"Final Val MSE: {val_mse:.6f}, Val MAE: {val_mae:.6f}")
    
    is_best = val_mae < best_mae
    if is_best:
        best_mae = val_mae
    
    # Save checkpoint in case we want to extend training
    save_checkpoint_supervised(optimizer, lr_scheduler, epochs-1, global_step, best_mae, training_history, save_dir)
    save_model(modnet,save_dir,is_best)
    
    # Save training history as JSON for plotting
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Models and training history saved in: {save_dir}")
    
def self_supervised_training(
        modnet:MODNet,
        file_path:str,
        device:str,
        batch_size:int=1,
        epochs:int=10,
        lr:float=0.00001,
        save_steps:int=10000,
        save_dir:str="./training/self-supervised",
        resume:bool=False
    ):
    os.makedirs(save_dir, exist_ok=True)
    train_dataloader = create_self_supervised_dataloader(file_path,batch_size)
    
    device = torch.device(device)
    blurer.to(device)
    modnet.to(device)
    optimizer = torch.optim.Adam(modnet.parameters(), lr=lr, betas=(0.9, 0.99))
    
    if resume:
        start_epoch, global_step, training_history = load_checkpoint_self_supervised(
            modnet, optimizer, save_dir
        )
    else:
        start_epoch = 0
        global_step = 0
        training_history = {'train_losses': []}
    
    stop_file = os.path.join(save_dir, 'STOP_TRAINING')
    assert not os.path.exists(stop_file), "The 'STOP_TRAINING' file must be deleted before training can be resumed."
    print(f"Begining training. To interrupt, create a file {stop_file}")
    
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epochs
    pbar = tqdm(initial=global_step, total=total_steps)
    losses = {
        'semantic_loss': [],
        'detail_loss': []
    }
    for epoch in range(start_epoch, epochs):
        backup_modnet = copy.deepcopy(modnet)
        
        steps_to_skip = 0
        if resume and epoch == start_epoch:
            steps_completed_in_epoch = global_step % steps_per_epoch
            steps_to_skip = steps_completed_in_epoch
            
        for image in train_dataloader:
            
            if steps_to_skip > 0:
                steps_to_skip -= 1
                continue
            
            image = image.to(device)
            
            semantic_loss, detail_loss = soc_adaptation_iter(modnet, backup_modnet, optimizer, image, device)
            
            losses['semantic_loss'].append(semantic_loss.item())
            losses['detail_loss'].append(detail_loss.item())
            
            global_step+=1
            pbar.update()
            total_loss = (semantic_loss + detail_loss).item()
            if device.type == 'cpu':
                pbar.set_description(f"Loss: {total_loss:.4f}")
            else:
                allocated = torch.cuda.memory_allocated() / 1024**2
                cached = torch.cuda.memory_reserved() / 1024**2
                pbar.set_description(f"Loss: {total_loss:.4f} | GPU: {allocated:.0f}MB/{cached:.0f}MB")
            
            if global_step % save_steps == 0:
                print(f"\nSaving at step {global_step}...")
                training_history['train_losses'].append({
                    'step': global_step,
                    'epoch': epoch,
                    'semantic_loss': sum(losses['semantic_loss']) / len(losses['semantic_loss']),
                    'detail_loss': sum(losses['detail_loss']) / len(losses['detail_loss'])
                })
                losses = {
                    'semantic_loss': [],
                    'detail_loss': []
                }
                
                save_model(modnet,save_dir,False)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                if os.path.exists(stop_file):
                    save_checkpoint_self_supervised(optimizer, epoch, global_step, training_history, save_dir)
                    pbar.close()
                    print(f"\nStop signal detected. Checkpoint saved.")
                    return
        
    pbar.close()
    
    print("\nTraining complete.")
    training_history['train_losses'].append({
        'step': global_step,
        'epoch': epoch,
        'semantic_loss': sum(losses['semantic_loss']) / len(losses['semantic_loss']),
        'detail_loss': sum(losses['detail_loss']) / len(losses['detail_loss'])
    })
    
    # Save checkpoint in case we want to extend training
    save_checkpoint_self_supervised(optimizer, epoch, global_step, training_history, save_dir)
    save_model(modnet,save_dir,False)
    
    # Save training history as JSON for plotting
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Model and training history saved in: {save_dir}")