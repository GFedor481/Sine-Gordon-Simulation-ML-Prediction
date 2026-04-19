import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import json

# Setup logging configuration for training progress tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==================== 2D Rotary Position Embedding (RoPE) ====================
class RotaryPositionalEmbedding2D(nn.Module):
    # [Unchanged: RoPE implementation remains exactly the same]
    def __init__(self, dim, max_seq_len_t=2048, max_seq_len_s=2048, theta=10000.0):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D RoPE (half for time, half for space)"
        
        self.dim = dim
        self.max_seq_len_t = max_seq_len_t
        self.max_seq_len_s = max_seq_len_s
        self.theta = theta

        self.dim_t = dim // 2  
        self.dim_s = dim // 2  

        inv_freq_t = 1.0 / (theta ** (torch.arange(0, self.dim_t, 2).float() / self.dim_t))
        self.register_buffer('inv_freq_t', inv_freq_t)

        inv_freq_s = 1.0 / (theta ** (torch.arange(0, self.dim_s, 2).float() / self.dim_s))
        self.register_buffer('inv_freq_s', inv_freq_s)
        
        self._cached_t = None
        self._cached_s = None
        self._sin_cached = None
        self._cos_cached = None
    
    def _update_cache(self, seq_len_t, seq_len_s, device):
        if seq_len_t != self._cached_t or seq_len_s != self._cached_s:
            self._cached_t = seq_len_t
            self._cached_s = seq_len_s
            
            t = torch.arange(seq_len_t, device=device).type_as(self.inv_freq_t)
            freqs_t = torch.einsum('i,j->ij', t, self.inv_freq_t)
            emb_t = torch.cat((freqs_t, freqs_t), dim=-1)
            
            s = torch.arange(seq_len_s, device=device).type_as(self.inv_freq_s)
            freqs_s = torch.einsum('i,j->ij', s, self.inv_freq_s)
            emb_s = torch.cat((freqs_s, freqs_s), dim=-1)
            
            emb_t = emb_t.unsqueeze(1).expand(-1, seq_len_s, -1)
            emb_s = emb_s.unsqueeze(0).expand(seq_len_t, -1, -1)
            
            emb = torch.cat([emb_t, emb_s], dim=-1)
            
            self._cos_cached = emb.cos()[None, :, :, None, :]
            self._sin_cached = emb.sin()[None, :, :, None, :]
        
        return self._cos_cached, self._sin_cached
    
    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, q, k, seq_len_t, seq_len_s, skip_first_n=0):
        cos, sin = self._update_cache(seq_len_t, seq_len_s, q.device)
        B, n_heads, L, head_dim = q.shape
        
        if skip_first_n > 0:
            q_sink = q[:, :, :skip_first_n, :]  
            k_sink = k[:, :, :skip_first_n, :]  
            q_main = q[:, :, skip_first_n:, :]  
            k_main = k[:, :, skip_first_n:, :]  
        else:
            q_main = q  
            k_main = k  
        
        cos = cos[:, :seq_len_t, :seq_len_s, :, :head_dim]  
        sin = sin[:, :seq_len_t, :seq_len_s, :, :head_dim]  
        
        cos = cos.reshape(1, 1, seq_len_t * seq_len_s, head_dim)  
        sin = sin.reshape(1, 1, seq_len_t * seq_len_s, head_dim)  
        
        q_embed = (q_main * cos) + (self.rotate_half(q_main) * sin)  
        k_embed = (k_main * cos) + (self.rotate_half(k_main) * sin)  
        
        if skip_first_n > 0:
            q_embed = torch.cat([q_sink, q_embed], dim=2)  
            k_embed = torch.cat([k_sink, k_embed], dim=2)  
        
        return q_embed, k_embed


# ==================== Soft Label Cross Entropy Loss ====================
class SoftLabelCrossEntropy(nn.Module):
    # [Unchanged]
    def __init__(self, num_classes=100, smoothing_sigma=3.0):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing_sigma = smoothing_sigma
    
    def forward(self, logits, targets):
        batch_size = logits.size(0)
        device = logits.device
        soft_targets = torch.zeros_like(logits)
        
        for i in range(batch_size):
            target = targets[i].item()  
            positions = torch.arange(self.num_classes, device=device).float()
            distances = torch.abs(positions - target)
            weights = torch.exp(-distances**2 / (2 * self.smoothing_sigma**2))
            soft_targets[i] = weights / weights.sum()
        
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        
        return loss


# ==================== Multi-Head Attention with RoPE ====================
class MultiHeadAttentionWithRoPE(nn.Module):
    # [Unchanged]
    def __init__(self, d_model, n_heads, dropout=0.1, max_seq_len_t=2048, max_seq_len_s=2048, use_rope=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D RoPE"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  
        self.use_rope = use_rope
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if use_rope:
            self.rope = RotaryPositionalEmbedding2D(
                self.head_dim, 
                max_seq_len_t=max_seq_len_t, 
                max_seq_len_s=max_seq_len_s
            )
        
    def forward(self, x, seq_len_t=None, seq_len_s=None, skip_first_n=0):
        B, L, D = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.use_rope:
            q, k = self.rope(q, k, seq_len_t, seq_len_s, skip_first_n=skip_first_n)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, L, D)
        out = self.out_proj(out)
        
        return out


# ==================== Transformer Encoder Layer ====================
class TransformerEncoderLayer(nn.Module):
    # [Unchanged]
    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1, 
                 max_seq_len_t=2048, max_seq_len_s=2048, use_rope=True):
        super().__init__()
        
        self.self_attn = MultiHeadAttentionWithRoPE(
            d_model, n_heads, dropout, max_seq_len_t, max_seq_len_s, use_rope
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),  
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, seq_len_t=None, seq_len_s=None, skip_first_n=0):
        x = x + self.self_attn(self.norm1(x), seq_len_t, seq_len_s, skip_first_n)
        x = x + self.ffn(self.norm2(x))
        return x


# ==================== Transformer Encoder ====================
class TransformerEncoder(nn.Module):
    # [Unchanged]
    def __init__(self, d_model, n_heads, n_layers, dim_feedforward, dropout=0.1, 
                 max_seq_len_t=2048, max_seq_len_s=2048, use_rope=True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, n_heads, dim_feedforward, dropout, 
                max_seq_len_t, max_seq_len_s, use_rope
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seq_len_t=None, seq_len_s=None, skip_first_n=0):
        for layer in self.layers:
            x = layer(x, seq_len_t, seq_len_s, skip_first_n)
        return self.norm(x)


# ==================== Main Model: Sine-Gordon Localization Transformer ====================
class SineGordonLocalizationTransformer(nn.Module):
    # [Updated docstrings and defaults for 2 features]
    def __init__(
        self,
        d_model=128,
        n_heads=4,
        n_layers=6,
        dim_feedforward=512,
        dropout=0.1,
        num_pendulums=100,
        num_timesteps=10,
        input_features=2,  # <--- UPDATED: Only theta and dtheta
        use_rope=True,
        use_attention_sink=True,
        num_sink_tokens=1
    ):
        super().__init__()
        
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D RoPE"
        
        self.d_model = d_model
        self.num_pendulums = num_pendulums
        self.num_timesteps = num_timesteps
        self.use_rope = use_rope
        self.use_attention_sink = use_attention_sink
        self.num_sink_tokens = num_sink_tokens
        
        self.input_proj = nn.Linear(input_features, d_model)
        
        if use_attention_sink:
            self.sink_tokens = nn.Parameter(torch.randn(1, num_sink_tokens, d_model) * 0.02)
        
        self.spatiotemporal_encoder = TransformerEncoder(
            d_model, n_heads, n_layers, dim_feedforward, dropout,
            max_seq_len_t=num_timesteps,
            max_seq_len_s=num_pendulums,
            use_rope=use_rope
        )
        
        self.temporal_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),  
            nn.Linear(d_model // 2, 1)  
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  
        )
    
    def forward(self, x):
        """
        Args:
            x: Input features, shape [B, T, P, 2]
               B = batch size
               T = number of time steps
               P = number of pendulums
        """
        B, T, P, num_feat = x.shape  
        x = self.input_proj(x)
        x = x.reshape(B, T * P, self.d_model)
        
        if self.use_attention_sink:
            sink = self.sink_tokens.expand(B, -1, -1)
            x = torch.cat([sink, x], dim=1)
            skip_first_n = self.num_sink_tokens
        else:
            skip_first_n = 0
        
        x = self.spatiotemporal_encoder(x, seq_len_t=T, seq_len_s=P, skip_first_n=skip_first_n)
        
        if self.use_attention_sink:
            x = x[:, self.num_sink_tokens:, :]
        
        x = x.reshape(B, T, P, self.d_model)
        x = x.permute(0, 2, 1, 3)
        
        attn_weights = self.temporal_pool(x)
        attn_weights = F.softmax(attn_weights, dim=2)
        
        x = (x * attn_weights).sum(dim=2)
        logits = self.classifier(x).squeeze(-1)
        
        return logits


# ==================== Dataset ====================
class SineGordonDataset(Dataset):
    # [Updated feature calculations to use only 2 features]
    def __init__(self, h5_file, split='train', train_ratio=0.7, val_ratio=0.15, seed=42):
        self.h5_file = h5_file
        self.split = split
        
        with h5py.File(h5_file, 'r') as hf:
            all_keys = [k for k in hf.keys() if k.startswith('simulation_')]
            
            np.random.seed(seed)
            np.random.shuffle(all_keys)
            
            n_total = len(all_keys)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            if split == 'train':
                self.sim_keys = all_keys[:n_train]
            elif split == 'val':
                self.sim_keys = all_keys[n_train:n_train+n_val]
            else:  
                self.sim_keys = all_keys[n_train+n_val:]
        
        logger.info(f"{split.upper()} dataset: {len(self.sim_keys)} samples")
        
        if split == 'train':
            self._compute_stats()
        
    def _compute_stats(self):
        logger.info("Computing dataset statistics for 2 features (theta, dtheta)...")
        thetas = []
        dthetas = []
        
        with h5py.File(self.h5_file, 'r') as hf:
            sample_keys = self.sim_keys[::max(1, len(self.sim_keys)//1000)][:1000]
            logger.info(f"Computing stats from {len(sample_keys)} samples...")
            
            for key in sample_keys:
                # Read theta and dtheta from HDF5
                theta = hf[key]['theta'][:]
                dtheta = hf[key]['dtheta'][:]
                
                thetas.append(theta)
                dthetas.append(dtheta)
        
        # Concatenate all samples
        thetas = np.concatenate(thetas, axis=0)
        dthetas = np.concatenate(dthetas, axis=0)
        
        theta_mean = thetas.mean()
        theta_std = thetas.std() + 1e-8
        dtheta_mean = dthetas.mean()
        dtheta_std = dthetas.std() + 1e-8
        
        # Store mean and std for only the 2 features: [theta, dtheta]
        self.mean = np.array([theta_mean, dtheta_mean])
        self.std = np.array([theta_std, dtheta_std])
        
        logger.info(f"Dataset mean: {self.mean}")
        logger.info(f"Dataset std: {self.std}")
    
    def __len__(self):
        return len(self.sim_keys)
    
    def __getitem__(self, idx):
        key = self.sim_keys[idx]
        
        with h5py.File(self.h5_file, 'r') as hf:
            sim_group = hf[key]
            
            # Load raw data from HDF5
            theta = sim_group['theta'][:]
            dtheta = sim_group['dtheta'][:]
            
            # Stack the 2 features along the last dimension
            # features shape: [T, P, 2]
            features = np.stack([theta, dtheta], axis=-1)
            
            loc_pendulum = sim_group.attrs['localized_pendulum']
            
            if hasattr(self, 'mean') and hasattr(self, 'std'):
                features = (features - self.mean) / self.std
        
        return torch.FloatTensor(features), torch.LongTensor([loc_pendulum])[0]


def collate_fn(batch):
    # [Unchanged except for the docstring reference]
    features, targets = zip(*batch)
    valid_indices = [i for i, t in enumerate(targets) if t >= 0]
    
    if len(valid_indices) == 0:
        return None, None
    
    features = torch.stack([features[i] for i in valid_indices])
    targets = torch.stack([targets[i] for i in valid_indices])
    
    return features, targets


# ==================== Training Functions ====================
def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, grad_clip=1.0):
    # [Unchanged]
    model.train()  
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc="Training")
    
    for features, targets in pbar:
        if features is None:
            continue
        
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(features)
                loss = criterion(logits, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    
    return total_loss / len(dataloader), 100. * correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    # [Unchanged]
    model.eval()  
    total_loss = 0
    correct = 0
    total = 0
    top5_correct = 0
    proximity_correct = 0
    tolerance1_correct = 0
    
    for features, targets in tqdm(dataloader, desc="Evaluating"):
        if features is None:
            continue
        
        features, targets = features.to(device), targets.to(device)
        logits = model(features)
        loss = criterion(logits, targets)
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        
        top2_pred = logits.topk(2, dim=1)[1]
        top5_correct += (top2_pred == targets.unsqueeze(1)).any(dim=1).sum().item()
        
        distance = torch.abs(pred - targets)
        proximity_correct += (distance < 5).sum().item()
        tolerance1_correct += (distance <= 1).sum().item()
        
        total += targets.size(0)
    
    metrics = {
        'loss': total_loss / len(dataloader),
        'top1_acc': 100. * correct / total,
        'top2_acc': 100. * top5_correct / total,
        'proximity_acc': 100. * proximity_correct / total,
        'tolerance1_acc': 100. * tolerance1_correct / total
    }
    
    return metrics


@torch.no_grad()
def test_and_log_predictions(model, dataloader, device, save_path='test_predictions.txt', num_samples=5):
    # [Unchanged]
    model.eval()  
    all_preds = []
    all_targets = []
    
    for features, targets in tqdm(dataloader, desc="Testing"):
        if features is None:
            continue
        
        features, targets = features.to(device), targets.to(device)
        logits = model(features)
        pred = logits.argmax(dim=1)
        
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    
    with open(save_path, 'w') as f:
        f.write(f"Total test samples: {len(all_targets)}\n")
        f.write(f"="*50 + "\n\n")
        
        for i in range(min(num_samples, len(all_targets))):
            msg = f"\nSample {i+1}/{num_samples}\n"
            msg += f"Target pendulum: {all_targets[i]}\n"
            msg += f"Predicted pendulum: {all_preds[i]}\n"
            msg += f"Error: {abs(all_preds[i] - all_targets[i])}\n"
            f.write(msg)
            logger.info(msg.strip())
    
    logger.info(f"Predictions saved to {save_path}")


def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    num_epochs,
    save_dir='checkpoints',
    early_stopping_patience=15,
    grad_clip=1.0
):
    # [Unchanged]
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    best_val_acc = 0
    patience_counter = 0
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, grad_clip
        )
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Val Top-1: {val_metrics['top1_acc']:.2f}%, "
                   f"Val Top-2: {val_metrics['top2_acc']:.2f}%, "
                   f"Val ±1: {val_metrics['tolerance1_acc']:.2f}%, "
                   f"Val Proximity: {val_metrics['proximity_acc']:.2f}%")
        
        if scheduler is not None:
            scheduler.step()
        
        if val_metrics['top1_acc'] > best_val_acc:
            best_val_acc = val_metrics['top1_acc']
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, save_dir / 'best_model_improved_1.pt')
            logger.info(f"Saved best model with val accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            logger.info(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")
            
            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered!")
                break
    
    logger.info("\n" + "="*50)
    logger.info("Testing best model...")
    checkpoint = torch.load(save_dir / 'best_model_improved_1.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}, "
               f"Test Top-1: {test_metrics['top1_acc']:.2f}%, "
               f"Test Top-2: {test_metrics['top2_acc']:.2f}%, "
               f"Test ±1: {test_metrics['tolerance1_acc']:.2f}%, "
               f"Test Proximity: {test_metrics['proximity_acc']:.2f}%")
    
    test_and_log_predictions(
        model, test_loader, device, 
        save_dir / 'test_predictions_improved.txt', 
        num_samples=5
    )
    
    with open(save_dir / 'test_results_improved.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    return test_metrics


# ==================== Main Training Script ====================
def main():
    # [Updated input_features config]
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    config = {
        'data_file': 'sine_gordon_dataset_200k.h5',
        'num_pendulums': 100,  
        'num_timesteps': 10,   
        'input_features': 2,  
        
        'd_model': 128,              
        'n_heads': 1,                
        'n_layers': 8,               
        'dim_feedforward': 128,      
        'dropout': 0.2,              
        'use_rope': True,            
        'use_attention_sink': True,  
        'num_sink_tokens': 2,      
        
        'batch_size': 64,                    
        'num_epochs': 128,                  
        'learning_rate': 1e-4,              
        'weight_decay': 1e-4,               
        'grad_clip': 5.0,                   
        'early_stopping_patience': 15,      
        
        'train_ratio': 0.45,   
        'val_ratio': 0.05,    
        
        'loss_type': 'soft_label',    
        'soft_label_sigma': 1.65,     
        
        'seed': 42,  
    }
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    logger.info("Loading datasets...")
    
    train_dataset = SineGordonDataset(
        config['data_file'], 
        split='train',
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        seed=config['seed']
    )
    
    val_dataset = SineGordonDataset(
        config['data_file'],
        split='val',
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        seed=config['seed']
    )
    
    test_dataset = SineGordonDataset(
        config['data_file'],
        split='test',
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        seed=config['seed']
    )
    
    val_dataset.mean = train_dataset.mean
    val_dataset.std = train_dataset.std
    test_dataset.mean = train_dataset.mean
    test_dataset.std = train_dataset.std
    
    logger.info("Creating dataloaders...")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4,  
        collate_fn=collate_fn,  
        pin_memory=True  
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4, 
        collate_fn=collate_fn, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4, 
        collate_fn=collate_fn, 
        pin_memory=True
    )
    
    logger.info("Initializing model...")
    
    model = SineGordonLocalizationTransformer(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        num_pendulums=config['num_pendulums'],
        num_timesteps=config['num_timesteps'],
        input_features=config['input_features'],
        use_rope=config['use_rope'],
        use_attention_sink=config['use_attention_sink'],
        num_sink_tokens=config['num_sink_tokens']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {n_params:,} trainable parameters")
    
    logger.info("Initializing optimizer and scheduler...")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],  
        eta_min=2e-7  
    )
    
    logger.info(f"Using loss function: {config['loss_type']}")
    
    criterion = SoftLabelCrossEntropy(
        num_classes=config['num_pendulums'],
        smoothing_sigma=config['soft_label_sigma']
    )
    
    Path('checkpoints').mkdir(exist_ok=True)
    with open('checkpoints/config_improved.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info("\nStarting training...")
    test_metrics = train(
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        scheduler,
        criterion,
        device,
        config['num_epochs'],
        save_dir='march_21_3',
        early_stopping_patience=config['early_stopping_patience'],
        grad_clip=config['grad_clip']
    )
    
    logger.info("\nTraining complete!")
    logger.info(f"Final test Top-1 accuracy: {test_metrics['top1_acc']:.2f}%")
    logger.info(f"Final test ±1 accuracy: {test_metrics['tolerance1_acc']:.2f}%")
    logger.info(f"Final test Top-2 accuracy: {test_metrics['top2_acc']:.2f}%")
    logger.info(f"Final test Proximity accuracy: {test_metrics['proximity_acc']:.2f}%")


if __name__ == '__main__':
    main()