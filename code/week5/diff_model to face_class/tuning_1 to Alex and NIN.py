import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
# 设置字体以支持中文（尝试多种常用中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from data_loader import get_data_loaders

# Import modified models
from AlexNet_modify import get_model as get_alexnet_mod
from NIN_modify import get_model as get_nin_mod

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = correct / total
        train_accs.append(train_acc)
        train_losses.append(running_loss / len(train_loader))
        
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        val_accs.append(val_acc)
        val_losses.append(val_running_loss / len(val_loader))
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
    total_time = time.time() - start_time
    return train_losses, train_accs, val_losses, val_accs, total_time

def plot_metrics(model_name, train_losses, val_losses, train_accs, val_accs, total_time, final_train_acc, final_val_acc, lr, batch_size, optimizer_name, save_dir='vis_results_tuning'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Text info
    info_text = (f"Model: {model_name}\n"
                 f"默认训练参数: LR={lr}, Batch={batch_size}, Opt={optimizer_name} | "
                 f"Time: {total_time:.2f}s | Train Acc: {final_train_acc:.4f} | Val Acc: {final_val_acc:.4f}")
    fig.suptitle(info_text, fontsize=12, fontweight='bold')
    
    # Plot Loss
    ax1.plot(train_losses, label='Training Loss', color='blue', marker='o')
    ax1.plot(val_losses, label='Validation Loss', color='red', marker='x')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Accuracy
    ax2.plot(train_accs, label='Training Accuracy', color='green', marker='o')
    ax2.plot(val_accs, label='Validation Accuracy', color='orange', marker='x')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.savefig(os.path.join(save_dir, f'{model_name}_metrics.png'), dpi=300)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = 'train_data'
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found.")
        return

    # Baseline Hyperparameters
    LR = 0.001
    BATCH_SIZE = 16
    EPOCHS = 10 # Reasonable default
    
    train_loader, val_loader, _ = get_data_loaders(data_dir, batch_size=BATCH_SIZE, split_ratio=0.8)
    
    # Only test modified models
    models_dict = {
        'AlexNet_modify': get_alexnet_mod,
        'NIN_modify': get_nin_mod
    }
    
    for model_name, model_fn in models_dict.items():
        print(f"\nTraining {model_name}...")
        torch.cuda.empty_cache()
        
        model = model_fn(num_classes=2).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        train_losses, train_accs, val_losses, val_accs, total_time = train_model(
            model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS, device=device
        )
        
        final_train_acc = train_accs[-1]
        final_val_acc = val_accs[-1]
        
        # Plot metrics
        plot_metrics(model_name, train_losses, val_losses, train_accs, val_accs, total_time, final_train_acc, final_val_acc, LR, BATCH_SIZE, 'Adam')
        
    print("\nTuning experiments completed.")

if __name__ == '__main__':
    main()
