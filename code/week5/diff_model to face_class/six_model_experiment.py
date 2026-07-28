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

# Import models
from LeNet import get_model as get_lenet
from AlexNet import get_model as get_alexnet
from VGG6 import get_model as get_vgg6
from NIN import get_model as get_nin
from Googlenet import get_model as get_googlenet
from resnet import get_model as get_resnet

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

def plot_metrics(model_name, train_losses, val_losses, train_accs, val_accs, total_time, final_train_acc, final_val_acc, lr, batch_size, optimizer_name, save_dir='vis_results'):
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
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.savefig(os.path.join(save_dir, f'{model_name}_metrics.png'), dpi=300)
    plt.close()

def visualize_feature_maps(model, image, model_name, device, save_dir='vis_results'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    model.eval()
    image = image.to(device).unsqueeze(0)
    
    feature_maps = {}
    hooks = []
    
    def get_hook(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach().cpu()
        return hook
        
    # Find all Conv2d layers and register hooks
    conv_idx = 1
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layer_name = f"{model_name}_Conv{conv_idx}"
            hooks.append(module.register_forward_hook(get_hook(layer_name)))
            conv_idx += 1
            
    if not hooks:
        print(f"No Conv2d layers found in {model_name}")
        return
    
    with torch.no_grad():
        model(image)
        
    for h in hooks:
        h.remove()
    
    # Plot and save for each layer
    for layer_name, fmap in feature_maps.items():
        # Plot first 16 channels (or less if fewer channels)
        num_channels = min(16, fmap.shape[1])
        rows = int(np.sqrt(num_channels))
        cols = int(np.ceil(num_channels / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        fig.suptitle(f'{layer_name} features')
        
        # Handle case where axes is not an array (single subplot)
        if num_channels == 1:
            axes = np.array([axes])
            
        for j in range(num_channels):
            if j < len(axes.flatten()):
                ax = axes.flatten()[j]
                ax.imshow(fmap[0, j], cmap='viridis')
                ax.axis('off')
            
        # Hide unused subplots
        for j in range(num_channels, len(axes.flatten())):
            axes.flatten()[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{layer_name}_features.png'))
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
    
    # Get a sample image for visualization
    sample_loader, _, _ = get_data_loaders(data_dir, batch_size=1, split_ratio=0.8)
    sample_image, _ = next(iter(sample_loader))
    sample_image = sample_image[0] # Remove batch dim
    
    models_dict = {
        'LeNet': get_lenet,
        'AlexNet': get_alexnet,
        'VGG6': get_vgg6,
        'NiN': get_nin,
        'GoogLeNet': get_googlenet,
        'ResNet': get_resnet
    }
    
    results = []
    
    for model_name, model_fn in models_dict.items():
        print(f"\nTraining {model_name} (Baseline)...")
        torch.cuda.empty_cache()
        
        if model_name == 'ResNet':
             model = model_fn(num_classes=2, dropout=0.5).to(device)
        else:
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
        
        # Visualize feature maps
        print(f"Visualizing feature maps for {model_name}...")
        visualize_feature_maps(model, sample_image, model_name, device)
        
        results.append({
            'Model_Name': model_name,
            'Total_Training_Time': total_time,
            'Final_Train_Acc': final_train_acc,
            'Final_Val_Acc': final_val_acc,
            'Optimizer': 'Adam',
            'Learning_Rate': LR,
            'Batch_Size': BATCH_SIZE,
            'Epochs': EPOCHS,
            'Split_Ratio': 0.8,
            'Image_Size': '100x100',
            'Dropout': 0.5
        })

    df = pd.DataFrame(results)
    df.to_csv('six_models_results.csv', index=False)
    print("\nBaseline experiments completed. Results saved to six_models_results.csv")

if __name__ == '__main__':
    main()
