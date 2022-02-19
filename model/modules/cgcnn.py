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
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1) # [N, atom_fea_len]
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
    Graph Embedding Layer for MOF Graph Transformer
    """

    def __init__(self, max_nbr_atoms, max_graph_len, hid_dim, nbr_fea_len):
        super().__init__()
        assert hid_dim % max_nbr_atoms == 0
        graph_emb_dim = hid_dim // max_nbr_atoms
        assert graph_emb_dim == nbr_fea_len

        self.node_embedding = nn.Embedding(119, graph_emb_dim)
        self.edge_embedding = nn.Embedding(119, graph_emb_dim)

        self.max_nbr_atoms = max_nbr_atoms
        self.max_graph_len = max_graph_len
        self.hid_dim = hid_dim

    def forward(self, atom_num, nbr_idx, nbr_fea, crystal_atom_idx, moc=None):
        """
        Args:
            atom_num (tensor): [N']
            nbr_idx (tensor): [N', M]
            nbr_fea (tensor): [N', M, nbr_fea_len]
            crystal_atom_idx (list): [B]
        Returns:
            new_atom_fea (tensor): [B, max_graph_len, hid_dim]
            mask (tensor): [B, max_graph_len]
        """

        batch_size = len(crystal_atom_idx)
        nbr_atom_num = atom_num[nbr_idx]

        emb_node = self.node_embedding(atom_num)[:, None, :].repeat(1, self.max_nbr_atoms, 1)  # [N', 64], [N', 12, 64]
        emb_edge = self.edge_embedding(nbr_atom_num)  # [N', 12, 64]
        emb_dist = nbr_fea  # [N', 12, 64]

        emb_total = emb_node + emb_edge + emb_dist  # [N', 12, 64]
        emb_total = emb_total.reshape([len(atom_num), -1])  # [N', 768]

        graph_emb = torch.zeros([batch_size, self.max_graph_len, self.hid_dim]).to(emb_total)
        # [B, max_graph_len, hid_dim]
        if moc:
            metalnode = moc  # [B]
            mo_label = torch.full([batch_size, self.max_graph_len], fill_value=-100).long()
        else:
            mo_label = None

        for bi, c_atom_idx in enumerate(crystal_atom_idx):
            if moc:
                mo = torch.zeros(len(c_atom_idx)).long()
                mo[torch.LongTensor(metalnode[bi])] = 1

            c_total_emb = emb_total[c_atom_idx]
            c_atom_num = atom_num[c_atom_idx]
            device_ = c_total_emb.device

            # others = non carbon and hydrogen
            mask_others = torch.logical_and(c_atom_num != 6, c_atom_num != 1)
            idx_others = torch.where(mask_others)[0]
            if len(idx_others) > 180:
                idx = torch.randperm(len(idx_others))[:180].long().to(device_)
                emb_others = c_total_emb[idx]
            else:
                idx = torch.randperm(len(idx_others)).long().to(device_)
                emb_others = torch.cat([c_total_emb[idx], torch.zeros([180 - len(idx), self.hid_dim]).to(device_)],
                                       dim=0)

            if moc:
                mo_label[bi, :len(idx)] = mo[idx]

            # carbon
            idx_carbon = torch.where(c_atom_num == 6)[0]
            if len(idx_carbon) > 120:
                idx = torch.randperm(len(idx_carbon))[:120].long().to(device_)
                emb_carbon = c_total_emb[idx]
            else:
                idx = torch.randperm(len(idx_carbon)).long().to(device_)
                emb_carbon = torch.cat([c_total_emb[idx], torch.zeros([120 - len(idx), self.hid_dim]).to(device_)],
                                       dim=0)

            if moc:
                mo_label[bi, 180:180 + len(idx)] = mo[idx]

            final_emb = torch.cat([emb_others, emb_carbon], dim=0)
            graph_emb[bi] = final_emb

        mask = (graph_emb.sum(dim=-1) != 0).float()

        return graph_emb, mask, mo_label


class GraphEmbeddings_Uni_Index(nn.Module):
    """
    Generate Embedding layers made by only convolution layers of CGCNN (not pooling)
    (https://github.com/txie-93/cgcnn)
    """

    def __init__(self, atom_fea_len, nbr_fea_len, max_graph_len, hid_dim, n_conv=3):
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

        new_atom_fea, mask = self.reconstruct_batch(atom_fea, crystal_atom_idx, uni_idx, uni_count)
        # [B, max_graph_len, hid_dim], [B, max_graph_len]
        return new_atom_fea, mask

    def reconstruct_batch(self, atom_fea, crystal_atom_idx, uni_idx, uni_count):
        batch_size = len(crystal_atom_idx)

        new_atom_fea = torch.full(
            size=[batch_size, self.max_graph_len, self.hid_dim],
            fill_value=0.
        ).to(atom_fea)

        for bi, c_atom_idx in enumerate(crystal_atom_idx):
            # set uni_idx with descending count and cut max_graph_len
            idx_ = torch.LongTensor(uni_idx[bi])
            count_ = torch.LongTensor(uni_count[bi])
            final_idx = idx_[torch.argsort(count_, descending=True)][:self.max_graph_len]
            new_atom_fea[bi][:len(final_idx)] = atom_fea[c_atom_idx][final_idx]
        mask = (new_atom_fea.sum(dim=-1) != 0).float()

        return new_atom_fea, mask
