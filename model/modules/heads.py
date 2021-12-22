import torch.nn as nn

from transformers.models.bert.modeling_bert import (
    BertConfig, BertPredictionHeadTransform
)


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
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
        self.decoder = nn.Linear(hid_dim, 101+2)  # bins

    def forward(self, x):  # [B, max_len, hid_dim]
        x = self.transform(x)  # [B, max_len, hid_dim]
        x = self.decoder(x)  # [B, max_len, bins]
        return x


class RegressionHead(nn.Module):
    """
    head for Regression
    """

    def __init__(self, hid_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(hid_dim)
        self.fc = nn.Linear(hid_dim, 1)

    def forward(self, x):
        x = self.bn(x)
        x = self.fc(x)
        return x


class ClassificationHead(nn.Module):
    """
    head for Classification
    """

    def __init__(self, hid_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.ln = nn.LayerNorm(hid_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
