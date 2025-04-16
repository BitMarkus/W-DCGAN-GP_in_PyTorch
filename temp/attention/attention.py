import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim//8, 1, bias=False)
        self.key = nn.Conv2d(in_dim, in_dim//8, 1, bias=False)
        self.value = nn.Conv2d(in_dim, in_dim, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Get query/key/value projections
        q = self.query(x).view(B, -1, H*W).permute(0, 2, 1)  # (B, HW, C')
        k = self.key(x).view(B, -1, H*W)  # (B, C', HW)
        v = self.value(x).view(B, -1, H*W)  # (B, C, HW)
        # Calculate attention
        attn = self.softmax(torch.bmm(q, k))  # (B, HW, HW)
        out = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C, HW)
        
        # Return output + residual
        return self.gamma * out.view(B, C, H, W) + x
    
    # Returns normalized attention heatmap for visualization
    def get_attention_map(self, x):

        with torch.no_grad():
            B, C, H, W = x.shape
            q = self.query(x).view(B, -1, H*W).permute(0, 2, 1)
            k = self.key(x).view(B, -1, H*W)
            # Calculate attention weights
            attn = self.softmax(torch.bmm(q, k))  # (B, HW, HW)
            # Average across batch and reshape to 2D
            attn_map = attn.mean(dim=0).mean(dim=0).view(H, W)
            # Normalize to [0, 1]
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            
            return attn_map.cpu().numpy()