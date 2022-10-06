import torch.nn as nn

from transformers.models.bert.modeling_bert import (
    BertConfig, BertPredictionHeadTransform
)


class Pooler(nn.Module):
    def __init__(self, hidden_size, index=0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.index = index

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, self.index]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GGMHead(nn.Module):
    """
    head for Graph Grid Matching
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MPPHead(nn.Module):
    """
    head for Masked Patch Prediction (regression version)
    """

    def __init__(self, hid_dim):
        super().__init__()

        bert_config = BertConfig(
            hidden_size=hid_dim,
        )
        self.transform = BertPredictionHeadTransform(bert_config)
        self.decoder = nn.Linear(hid_dim, 101 + 2)  # bins

    def forward(self, x):  # [B, max_len, hid_dim]
        x = self.transform(x)  # [B, max_len, hid_dim]
        x = self.decoder(x)  # [B, max_len, bins]
        return x


class MTPHead(nn.Module):
    """
    head for MOF Topology Prediction
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.fc = nn.Linear(hid_dim, 1100)  # num_topology : 1100

    def forward(self, x):
        x = self.fc(x)
        return x


class VFPHead(nn.Module):
    """
    head for Void Fraction Prediction
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(hid_dim)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = self.bn(x)
        x = self.fc(x)
        return x


class RegressionHead(nn.Module):
    """
    head for Regression
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


class ClassificationHead(nn.Module):
    """
    head for Classification
    """

    def __init__(self, hid_dim, n_classes):
        super().__init__()

        if n_classes == 2:
            self.fc = nn.Linear(hid_dim, 1)
            self.binary = True
        else:
            self.fc = nn.Linear(hid_dim, n_classes)
            self.binary = False

    def forward(self, x):
        x = self.fc(x)

        return x, self.binary


class MOCHead(nn.Module):
    """
    head for Metal Organic Classification
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        """
        :param x: graph_feats [B, graph_len, hid_dim]
        :return: [B, graph_len]
        """
        x = self.fc(x)  # [B, graph_len, 1]
        x = x.squeeze(dim=-1)  # [B, graph_len]
        return x


class BBPHead(nn.Module):
    """
    head for Building Block Prediction
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.fc = nn.Linear(hid_dim, 862)

    def forward(self, x):
        """
        :param x: output [B, hid_dim]
        :return: [B, num_bbs]
        """
        x = self.fc(x)
        return x
