import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
import numpy as np

# Configuration (kept your original config, but using a dict for more Pythonic style)
cfg = {
    'embed_dim': 16,
    'sparse_features': {
        'user_id': 1000000,
        'item_id': 500000,
        'category_id': 5000
    },
    'dense_dim': 8,
    'mlp_dims': [256, 128, 64],
    'dropout': 0.2
}

# Simple dataset class (assumes data is dict of tensors; replace with your actual data)
class RecDataset(Dataset):
    def __init__(self, sparse_data, dense_data, labels):
        self.sparse = sparse_data  # dict of tensors
        self.dense = dense_data    # tensor
        self.labels = labels       # tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'sparse': {k: v[idx] for k, v in self.sparse.items()},
            'dense': self.dense[idx],
            'label': self.labels[idx]
        }

# DeepFM model (your original model, slightly optimized)
class DeepFM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(num_embeddings, cfg['embed_dim'])
            for feat, num_embeddings in cfg['sparse_features'].items()
        })
        self.fm_linear = nn.Linear(len(cfg['sparse_features']) * cfg['embed_dim'] + cfg['dense_dim'], 1)
        self.mlp = nn.Sequential(
            nn.Linear(len(cfg['sparse_features']) * cfg['embed_dim'] + cfg['dense_dim'], cfg['mlp_dims'][0]),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            *[nn.Sequential(nn.Linear(cfg['mlp_dims'][i], cfg['mlp_dims'][i+1]), nn.ReLU(), nn.Dropout(cfg['dropout']))
              for i in range(len(cfg['mlp_dims']) - 1)],
            nn.Linear(cfg['mlp_dims'][-1], 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, sparse_inputs, dense_inputs):
        embeds = [self.embeddings[feat](sparse_inputs[feat]) for feat in sparse_inputs]
        embed_concat = torch.cat(embeds, dim=1)
        all_inputs = torch.cat([embed_concat, dense_inputs], dim=1)
        
        # FM part (simplified; add full FM if needed)
        fm_out = self.fm_linear(all_inputs)
        
        # Deep part
        mlp_out = self.mlp(all_inputs)
        
        return self.sigmoid(fm_out + mlp_out).squeeze(1)

# Example data (replace with real data, such as Criteo dataset)
sparse_data = {'user_id': torch.randint(0, cfg['sparse_features']['user_id'], (1000,)),
               'item_id': torch.randint(0, cfg['sparse_features']['item_id'], (1000,)),
               'category_id': torch.randint(0, cfg['sparse_features']['category_id'], (1000,))}
dense_data = torch.randn(1000, cfg['dense_dim'])
labels = torch.randint(0, 2, (1000,)).float()

dataset = RecDataset(sparse_data, dense_data, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepFM(cfg).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

epochs = 10
all_preds, all_labels = [], []
for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        sparse = {k: v.to(device) for k, v in batch['sparse'].items()}
        dense = batch['dense'].to(device)
        label = batch['label'].to(device)
        
        optimizer.zero_grad()
        y_pred = model(sparse, dense)
        loss = criterion(y_pred, label)
        loss.backward()
        optimizer.step()
    
    # Evaluation (added AUC)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            sparse = {k: v.to(device) for k, v in batch['sparse'].items()}
            dense = batch['dense'].to(device)
            y_pred = model(sparse, dense).cpu().numpy()
            all_preds.extend(y_pred)
            all_labels.extend(batch['label'].numpy())
    auc = roc_auc_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}: AUC = {auc:.4f}")

# Save model (hint for productionization)
torch.save(model.state_dict(), 'deepfm_model.pth')
