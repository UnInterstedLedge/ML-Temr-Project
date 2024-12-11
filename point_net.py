import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Depth-Aware Dropout
class DepthAwareDropout(nn.Module):
    def __init__(self, base_p=0.3, max_p=0.7):
        """
        A custom dropout layer where the dropout probability varies based on depth.
        :param base_p: Minimum dropout probability.
        :param max_p: Maximum dropout probability for farthest points.
        """
        super(DepthAwareDropout, self).__init__()
        self.base_p = base_p
        self.max_p = max_p

    def forward(self, x, depths):
        """
        :param x: Input features [B, C, N].
        :param depths: Depth values normalized between 0 and 1 [B, N].
        """
        batch_size, _, num_points = x.size()

        # Expand normalized depths to match feature dimensions
        depths = depths.unsqueeze(1).expand(batch_size, x.size(1), num_points)

        # Calculate depth aware dropout probabilities threshold
        dropout_probs = self.base_p + (self.max_p - self.base_p) * depths

        # Create a dropout mask based on probabilities
        random_tensor = torch.rand_like(x)
        dropout_mask = random_tensor > dropout_probs

        # Apply dropout
        return x * dropout_mask.float()

# ============================================================================
# T-net (Spatial Transformer Network)
class Tnet(nn.Module):
    def __init__(self, dim, num_points=2500):
        super(Tnet, self).__init__()

        self.dim = dim
        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)

    def forward(self, x):
        bs = x.shape[0]
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.max_pool(x).view(bs, -1)
        x = self.bn4(F.relu(self.linear1(x)))
        x = self.bn5(F.relu(self.linear2(x)))
        x = self.linear3(x)
        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x.view(-1, self.dim, self.dim) + iden
        return x

# ============================================================================
# PointNet Backbone
class PointNetBackbone(nn.Module):
    def __init__(self, num_points=2500, num_global_feats=1024, local_feat=True):
        super(PointNetBackbone, self).__init__()
        self.num_points = num_points
        self.num_global_feats = num_global_feats
        self.local_feat = local_feat
        self.tnet1 = Tnet(dim=3, num_points=num_points)
        self.tnet2 = Tnet(dim=64, num_points=num_points)
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, num_global_feats, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(num_global_feats)
        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)

    def forward(self, x):
        bs = x.shape[0]
        A_input = self.tnet1(x)
        x = torch.bmm(x.transpose(2, 1), A_input).transpose(2, 1)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        A_feat = self.tnet2(x)
        x = torch.bmm(x.transpose(2, 1), A_feat).transpose(2, 1)
        local_features = x.clone()
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))
        global_features, critical_indexes = self.max_pool(x)
        global_features = global_features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)
        if self.local_feat:
            features = torch.cat((local_features, global_features.unsqueeze(-1).repeat(1, 1, self.num_points)), dim=1)
            return features, critical_indexes, A_feat
        else:
            return global_features, critical_indexes, A_feat

# ============================================================================
# Classification Head
class PointNetClassHead(nn.Module):
    def __init__(self, num_points=2500, num_global_feats=1024, k=2):
        super(PointNetClassHead, self).__init__()
        self.backbone = PointNetBackbone(num_points, num_global_feats, local_feat=False)
        self.linear1 = nn.Linear(num_global_feats, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.depth_aware_dropout = DepthAwareDropout(base_p=0.3, max_p=0.7)

    def forward(self, x, depths):
        x, crit_idxs, A_feat = self.backbone(x)
        x = self.bn1(F.relu(self.linear1(x)))
        x = self.depth_aware_dropout(x.unsqueeze(-1), depths).squeeze(-1)
        x = self.bn2(F.relu(self.linear2(x)))
        x = self.linear3(x)
        return x, crit_idxs, A_feat

# ============================================================================
# Segmentation Head
class PointNetSegHead(nn.Module):
    def __init__(self, num_points=2500, num_global_feats=1024, m=2):
        super(PointNetSegHead, self).__init__()
        self.num_points = num_points
        self.m = m
        self.backbone = PointNetBackbone(num_points, num_global_feats, local_feat=True)
        self.conv1 = nn.Conv1d(num_global_feats + 64, 512, kernel_size=1)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=1)
        self.conv4 = nn.Conv1d(128, m, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.depth_aware_dropout = DepthAwareDropout(base_p=0.3, max_p=0.7)

    def forward(self, x, depths):
        x, crit_idxs, A_feat = self.backbone(x)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.depth_aware_dropout(x, depths)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1)
        return x, crit_idxs, A_feat


#if __name__ == '__main__':
#    main()

