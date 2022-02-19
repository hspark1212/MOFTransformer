import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from model.modules import objectives, heads, module_utils
from model.modules.cgcnn import GraphEmbeddings, CrystalGraphConvNet, GraphEmbeddings_Uni_Index
from model.modules.egcnn import generate_resnet_model
from model.modules.vision_transformer_3d import VisionTransformer3D

from model.modules.module_utils import Normalizer

from torchmetrics.functional import r2_score

class Module(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.use_cgcnn = config["use_cgcnn"]
        self.use_egcnn = config["use_egcnn"]
        self.use_transformer = config["use_transformer"]

        self.max_grid_len = config["max_grid_len"]
        self.strategy = config["strategy"]

        if self.use_cgcnn:
            self.cgcnn = CrystalGraphConvNet(
                atom_fea_len=config["atom_fea_len"],
                nbr_fea_len=config["nbr_fea_len"],
                n_conv=config["n_conv"],
                hid_dim=config["hid_dim"],
            )
            self.cgcnn.apply(objectives.init_weights)

        if self.use_egcnn:
            self.egcnn = generate_resnet_model(
                model_depth=config["egcnn_depth"],
                n_input_channels=1,
                n_classes=config["hid_dim"],
            )
            self.egcnn.apply(objectives.init_weights)

        if self.use_transformer:
            self.graph_embeddings = GraphEmbeddings(
                max_nbr_atoms=config["max_nbr_atoms"],
                max_graph_len=config["max_graph_len"],
                hid_dim=config["hid_dim"],
                nbr_fea_len=config["nbr_fea_len"],
            )
            self.graph_embeddings.apply(objectives.init_weights)

            # token type embeddings
            self.token_type_embeddings = nn.Embedding(2, config["hid_dim"])
            self.token_type_embeddings.apply(objectives.init_weights)

            # set transformer
            self.transformer = VisionTransformer3D(
                img_size=config["img_size"],
                patch_size=config["patch_size"],
                in_chans=config["in_chans"],
                embed_dim=config["hid_dim"],
                depth=config["num_layers"],
                num_heads=config["num_heads"],
                mlp_ratio=config["mlp_ratio"],
                drop_rate=config["drop_rate"],
                mpp_ratio=config["mpp_ratio"],
            )

            # volume token
            self.volume_embeddings = nn.Linear(1, config["hid_dim"])
            self.volume_embeddings.apply(objectives.init_weights)

            # pooler
            self.pooler = heads.Pooler(config["hid_dim"])
            self.pooler.apply(objectives.init_weights)

        self.use_only_vit = config["use_only_vit"]
        if self.use_only_vit:
            """future removed..."""
            # set vision transformer
            self.transformer = VisionTransformer3D(
                img_size=config["img_size"],
                patch_size=config["patch_size"],
                in_chans=config["in_chans"],
                embed_dim=config["hid_dim"],
                depth=config["num_layers"],
                num_heads=config["num_heads"],
                mlp_ratio=config["mlp_ratio"],
                drop_rate=config["drop_rate"],
                mpp_ratio=config["mpp_ratio"],
            )

            # volume token
            self.volume_embeddings = nn.Linear(1, config["hid_dim"])
            self.volume_embeddings.apply(objectives.init_weights)

            # pooler
            self.pooler = heads.Pooler(config["hid_dim"])
            self.pooler.apply(objectives.init_weights)

        self.use_only_mgt = config["use_only_mgt"]
        if self.use_only_mgt:
            """future removed..."""
            self.graph_embeddings = GraphEmbeddings(
                max_nbr_atoms=config["max_nbr_atoms"],
                max_graph_len=config["max_graph_len"],
                hid_dim=config["hid_dim"],
                nbr_fea_len=config["nbr_fea_len"],
            )
            self.graph_embeddings.apply(objectives.init_weights)

            self.transformer = VisionTransformer3D(
                img_size=config["img_size"],
                patch_size=config["patch_size"],
                in_chans=config["in_chans"],
                embed_dim=config["hid_dim"],
                depth=config["num_layers"],
                num_heads=config["num_heads"],
                mlp_ratio=config["mlp_ratio"],
                drop_rate=config["drop_rate"],
                mpp_ratio=config["mpp_ratio"],
            )

            # pooler
            self.pooler = heads.Pooler(config["hid_dim"])
            self.pooler.apply(objectives.init_weights)

        # ===================== loss =====================
        if config["loss_names"]["ggm"] > 0:
            self.ggm_head = heads.GGMHead(config["hid_dim"])
            self.ggm_head.apply(objectives.init_weights)

        if config["loss_names"]["mpp"] > 0:
            self.mpp_head = heads.MPPHead(config["hid_dim"])
            self.mpp_head.apply(objectives.init_weights)

        if config["loss_names"]["mtp"] > 0:
            self.mtp_head = heads.MTPHead(config["hid_dim"])
            self.mtp_head.apply(objectives.init_weights)

        if config["loss_names"]["vfp"] > 0:
            self.vfp_head = heads.VFPHead(config["hid_dim"])
            self.vfp_head.apply(objectives.init_weights)

        if config["loss_names"]["moc"] > 0:
            self.moc_head = heads.MOCHead(config["hid_dim"])
            self.moc_head.apply(objectives.init_weights)

        # ===================== Downstream =====================
        if config["load_path"] != "" and not config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

        if self.use_cgcnn and self.use_egcnn and self.strategy == 'concat':
            # concat
            hid_dim = config["hid_dim"] * 2
        else:
            hid_dim = config["hid_dim"]

        if self.hparams.config["loss_names"]["regression"] > 0:
            self.regression_head = heads.RegressionHead(hid_dim)
            self.regression_head.apply(objectives.init_weights)
            # normalization
            self.mean = config["mean"]
            self.std = config["std"]


        if self.hparams.config["loss_names"]["classification"] > 0:
            n_classes = config["n_classes"]
            self.classification_head = heads.ClassificationHead(hid_dim, n_classes)
            self.classification_head.apply(objectives.init_weights)

        module_utils.set_metrics(self)
        self.current_tasks = list()
        # ===================== load downstream (test_only) ======================

        if config["load_path"] != "" and config["test_only"]:
            ckpt = torch.load(config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            print(f"load model : {config['load_path']}")

    def infer(self,
              batch,
              mask_grid=False,
              ):

        cif_id = batch["cif_id"]
        atom_num = batch["atom_num"]  # [N']
        nbr_idx = batch["nbr_idx"]  # [N', M]
        nbr_fea = batch["nbr_fea"]  # [N', M, nbr_fea_len]
        crystal_atom_idx = batch["crystal_atom_idx"]  # list [B]
        # uni_idx = batch["uni_idx"]  # list [B]
        # uni_count = batch["uni_count"]  # list [B]

        grid = batch["grid"]  # [B, C, H, W, D]
        volume = batch["volume"]  # list [B]

        moc = batch.get("moc") # if moc, [B]

        if self.use_cgcnn and self.use_egcnn:

            out_cgcnn = self.cgcnn(
                atom_num=atom_num,
                nbr_fea=nbr_fea,
                nbr_fea_idx=nbr_idx,
                crystal_atom_idx=crystal_atom_idx,
            )  # [B,hid_dim]
            out_egcnn = self.egcnn(grid)  # [B, hid_dim]

            if self.strategy == 'concat':
                out = torch.cat([out_cgcnn, out_egcnn], dim=-1)  # [B, hid_dim*2]
            elif self.strategy == 'element_wise_sum':
                out = out_cgcnn + out_egcnn
            elif self.strategy == 'element_wise_multiplication':
                out = torch.mul(out_cgcnn, out_egcnn)
            else:
                raise ValueError(
                    f'Strategy must be concat, element_wise_sum, or element_wise_multiplication, not {self.strategy}')

            ret = {
                "output": out
            }
            return ret

        elif self.use_cgcnn and not self.use_egcnn:

            out_cgcnn = self.cgcnn(
                atom_num=atom_num,
                nbr_fea=nbr_fea,
                nbr_fea_idx=nbr_idx,
                crystal_atom_idx=crystal_atom_idx,
            )  # [B,hid_dim]

            ret = {
                "output": out_cgcnn,
            }
            return ret

        elif not self.use_cgcnn and self.use_egcnn:

            out_egcnn = self.egcnn(grid)  # [B, hid_dim]
            ret = {
                "output": out_egcnn,
            }
            return ret

        elif self.use_transformer:

            (graph_embeds,  # [B, max_graph_len, hid_dim],
             graph_masks,  # [B, max_graph_len],
             mo_labels, # if moc: [B, max_graph_len], else: None
             ) = self.graph_embeddings(
                atom_num=atom_num,
                nbr_idx=nbr_idx,
                nbr_fea=nbr_fea,
                crystal_atom_idx=crystal_atom_idx,
                moc=moc,
            )

            (grid_embeds,  # [B, max_grid_len+1, hid_dim]
             grid_masks,  # [B, max_grid_len+1]
             grid_labels,  # [B, grid+1, C] if mask_image == True
             ) = self.transformer.visual_embed(
                grid,
                max_image_len=self.max_grid_len,
                mask_it=mask_grid,
            )
            # add volume embeds to grid_embeds
            volume = torch.FloatTensor(volume).to(grid_embeds)  # [B]
            volume_embeds = self.volume_embeddings(volume[:, None, None])  # [B, 1, hid_dim]
            volume_mask = torch.ones(volume.shape[0], 1).to(grid_masks)

            grid_embeds = torch.cat([grid_embeds, volume_embeds], dim=1)  # [B, max_grid_len+2, hid_dim]
            grid_masks = torch.cat([grid_masks, volume_mask], dim=1)  # [B, max_grid_len+2, hid_dim]

            # add token_type_embeddings
            graph_embeds = graph_embeds \
                           + self.token_type_embeddings(torch.zeros_like(graph_masks, device=self.device).long())
            grid_embeds = grid_embeds \
                          + self.token_type_embeddings(torch.ones_like(grid_masks, device=self.device).long())

            co_embeds = torch.cat([graph_embeds, grid_embeds], dim=1)  # [B, final_max_len, hid_dim]
            co_masks = torch.cat([graph_masks, grid_masks], dim=1)  # [B, final_max_len, hid_dim]

            x = co_embeds

            for i, blk in enumerate(self.transformer.blocks):
                x, _attn = blk(x, mask=co_masks)

            x = self.transformer.norm(x)
            graph_feats, grid_feats = (
                x[:, :graph_embeds.shape[1]],
                x[:, graph_embeds.shape[1]:],
            )  # [B, max_graph_len, hid_dim], [B, max_grid_len+2, hid_dim]

            cls_feats = self.pooler(x)  # [B, hid_dim]

            ret = {
                "graph_feats": graph_feats,
                "grid_feats": grid_feats,
                "output": cls_feats,
                "raw_cls_feats": x[:, 0],
                "graph_masks": graph_masks,
                "grid_masks": grid_masks,
                "grid_labels": grid_labels,  # if MPP, else None
                "mo_labels": mo_labels, # if MOC, else None
                "cif_id": cif_id,
            }

            return ret

        elif self.use_only_vit:
            """future removed..."""
            (grid_embeds,  # [B, max_grid_len+1, hid_dim]
             grid_masks,  # [B, max_grid_len+1]
             grid_labels,  # [B, grid+1, C] if mask_image == True
             ) = self.transformer.visual_embed(
                grid,
                max_image_len=self.max_grid_len,
                mask_it=mask_grid,
            )

            # volume embeds
            volume = torch.FloatTensor(volume).to(grid_embeds)  # [B]
            volume_embeds = self.volume_embeddings(volume[:, None, None])  # [B, 1, hid_dim]
            volume_mask = torch.ones(volume.shape[0], 1).to(grid_masks)

            grid_embeds = torch.cat([grid_embeds, volume_embeds], dim=1)
            grid_masks = torch.cat([grid_masks, volume_mask], dim=1)

            x = grid_embeds
            for i, blk in enumerate(self.transformer.blocks):
                x, _attn = blk(x, mask=grid_masks)

            x = self.transformer.norm(x)
            grid_feats = x

            cls_feats = self.pooler(x)  # [B, hid_dim]

            ret = {
                "grid_feats": grid_feats,
                "output": cls_feats,
                "raw_cls_feats": x[:, 0],
                "grid_masks": grid_masks,
                "grid_labels": grid_labels,  # if MPP, else None
            }

            return ret

        elif self.use_only_mgt:

            (graph_embeds,  # [B, max_graph_len, hid_dim],
             graph_masks,  # [B, max_graph_len],
             mo_labels, # if moc: [B, max_graph_len], else: None
             ) = self.graph_embeddings(
                atom_num=atom_num,
                nbr_idx=nbr_idx,
                nbr_fea=nbr_fea,
                crystal_atom_idx=crystal_atom_idx,
                moc=moc,
            )

            x = graph_embeds

            for i, blk in enumerate(self.transformer.blocks):
                x, _attn = blk(x, mask=graph_masks)

            x = self.transformer.norm(x)
            graph_feats = x

            cls_feats = self.pooler(x)  # [B, hid_dim]

            ret = {
                "graph_feats": graph_feats,
                "output": cls_feats,
                "graph_masks": graph_masks,
                "mo_labels": mo_labels,  # if MOC, else None
            }

            return ret

    def forward(self, batch):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Graph Grid Matching
        if "ggm" in self.current_tasks:
            ret.update(objectives.compute_ggm(self, batch))

        # MOF Topology Prediction
        if "mtp" in self.current_tasks:
            ret.update(objectives.compute_mtp(self, batch))

        # Void Fraction Prediction
        if "vfp" in self.current_tasks:
            ret.update(objectives.compute_vfp(self, batch))

        # Metal Organic Classification
        if "moc" in self.current_tasks:
            ret.update(objectives.compute_moc(self, batch))

        # regression
        if "regression" in self.current_tasks:
            normalizer = Normalizer(self.mean, self.std)
            ret.update(objectives.compute_regression(self, batch, normalizer))

        # classification
        if "classification" in self.current_tasks:
            ret.update(objectives.compute_classification(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        module_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])

        return total_loss

    def training_epoch_end(self, outputs):
        module_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        module_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outputs):
        module_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        module_utils.set_task(self)
        output = self(batch)
        return output

    def test_epoch_end(self, outputs):
        module_utils.epoch_wrapup(self)

        # calculate r2 score when regression
        if "regression_logits" in outputs[0].keys():
            logits = []
            labels = []
            for out in outputs:
                logits += out["regression_logits"].tolist()
                labels += out["regression_labels"].tolist()
            r2 = r2_score(torch.FloatTensor(logits), torch.FloatTensor(labels))
            self.log(f"test/r2_score", r2)



    def configure_optimizers(self):
        return module_utils.set_schedule(self)
