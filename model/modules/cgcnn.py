import random

import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    (https://github.com/txie-93/cgcnn)
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
                                 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Args:
        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns:

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """

        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]  # [N, M, atom_fea_len]

        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             # [N, atom_fea_len] -> [N, M, atom_fea_len] -> v_i
             atom_nbr_fea,  # [N, M, atom_fea_len] -> v_j
             nbr_fea],  # [N, M, nbr_fea_len] -> u(i,j)_k
            dim=2)
        # [N, M, atom_fea_len*2+nrb_fea_len]

        total_gated_fea = self.fc_full(total_nbr_fea)  # [N, M, atom_fea_len*2]
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len * 2)).view(N, M, self.atom_fea_len * 2)  # [N, M, atom_fea_len*2]
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)  # [N, M, atom_fea_len]
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)  # [N, atom_fea_len]
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)  # [N, atom_fea_len]
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    (https://github.com/txie-93/cgcnn)
    """

    def __init__(self, atom_fea_len, nbr_fea_len, hid_dim,
                 n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        nbr_fea_len: int
          Number of bond features.
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Embedding(119, atom_fea_len)  # 119 -> max(atomic number)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                              nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h - 1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h - 1)])

        self.fc_out = nn.Linear(h_fea_len, hid_dim)

    def forward(self, atom_num, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_num: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution

        """
        atom_fea = self.embedding(atom_num)  # [N, atom_fea_len]
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)  # [N, atom_fea_len]
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)  # [N0, atom_fea_len]
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))  # [N0, h_fea_len]
        crys_fea = self.conv_to_fc_softplus(crys_fea)  # [N0, h_fea_len]
        out = self.fc_out(crys_fea)

        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        """
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == \
               atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)


class GraphEmbeddings(nn.Module):
    """
    Generate Embedding layers made by only convolution layers of CGCNN (not pooling)
    (https://github.com/txie-93/cgcnn)
    """

    def __init__(self, atom_fea_len, nbr_fea_len, max_graph_len, hid_dim, n_conv=3, vis=False):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.max_graph_len = max_graph_len
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(119, atom_fea_len)  # 119 -> max(atomic number)
        self.convs = nn.ModuleList(
            [
                ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
                for _ in range(n_conv)
            ]
        )
        self.fc = nn.Linear(atom_fea_len, hid_dim)

        self.vis = vis

    def forward(self, atom_num, nbr_idx, nbr_fea, crystal_atom_idx, uni_idx, uni_count, moc=None):
        """
        Args:
            atom_num (tensor): [N', atom_fea_len]
            nbr_idx (tensor): [N', M]
            nbr_fea (tensor): [N', M, nbr_fea_len]
            crystal_atom_idx (list): [B]
            uni_idx (list) : [B]
            uni_count (list) : [B]
        Returns:
            new_atom_fea (tensor): [B, max_graph_len, hid_dim]
            mask (tensor): [B, max_graph_len]
        """
        assert self.nbr_fea_len == nbr_fea.shape[-1]

        atom_fea = self.embedding(atom_num)  # [N', atom_fea_len]
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_idx)  # [N', atom_fea_len]
        atom_fea = self.fc(atom_fea)  # [N', hid_dim]

        new_atom_fea, mask, mo_label = self.reconstruct_batch(atom_fea, crystal_atom_idx, uni_idx, uni_count, moc)
        # [B, max_graph_len, hid_dim], [B, max_graph_len]
        return new_atom_fea, mask, mo_label  # None will be replaced with MOC

    def reconstruct_batch(self, atom_fea, crystal_atom_idx, uni_idx, uni_count, moc):
        batch_size = len(crystal_atom_idx)

        new_atom_fea = torch.full(
            size=[batch_size, self.max_graph_len, self.hid_dim],
            fill_value=0.
        ).to(atom_fea)

        mo_label = torch.full(
            size=[batch_size, self.max_graph_len],
            fill_value=-100.
        ).to(atom_fea)

        for bi, c_atom_idx in enumerate(crystal_atom_idx):
            # set uni_idx with (descending count or random) and cut max_graph_len
            idx_ = torch.LongTensor([random.choice(u) for u in uni_idx[bi]])[:self.max_graph_len]
            rand_idx = idx_[torch.randperm(len(idx_))]
            if self.vis:
                rand_idx = idx_
            new_atom_fea[bi][:len(rand_idx)] = atom_fea[c_atom_idx][rand_idx]

            if moc:
                mo = torch.zeros(len(c_atom_idx))
                metal_idx = moc[bi]
                mo[metal_idx] = 1
                mo_label[bi][:len(rand_idx)] = mo[rand_idx]

        mask = (new_atom_fea.sum(dim=-1) != 0).float()

        return new_atom_fea, mask, mo_label
