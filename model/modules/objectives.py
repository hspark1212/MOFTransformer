import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import mean_absolute_error


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def compute_regression(pl_module, batch, normalizer):
    infer = pl_module.infer(batch, mask_grid=False)

    logits = pl_module.regression_head(infer["cls_feats"]).squeeze(-1)  # [B]
    labels = torch.FloatTensor(batch["target"]).to(logits.device)  # [B]
    assert len(labels.shape) == 1

    # normalize encode if config["mean"] and config["std], else pass
    labels = normalizer.encode(labels)

    loss = F.mse_loss(logits, labels)
    ret = {
        "regression_loss": loss,
        "regression_logits": normalizer.decode(logits),
        "regression_labels": normalizer.decode(labels),
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_regression_loss")(ret["regression_loss"])
    mae = getattr(pl_module, f"{phase}_regression_mae")(
        mean_absolute_error(ret["regression_logits"], ret["regression_labels"])
    )

    pl_module.log(f"regression/{phase}/loss", loss)
    pl_module.log(f"regression/{phase}/mae", mae)

    return ret


def compute_classification(pl_module, batch):
    infer = pl_module.infer(batch, mask_grid=False)

    logits, binary = pl_module.classification_head(infer["cls_feats"])  # [B, output_dim]
    labels = torch.LongTensor(batch["target"]).to(logits.device)  # [B]
    assert len(labels.shape) == 1
    if binary:
        logits = logits.squeeze(dim=-1)
        loss = F.binary_cross_entropy_with_logits(input=logits, target=labels.float())
    else:
        loss = F.cross_entropy(logits, labels)

    ret = {
        "classification_loss": loss,
        "classification_logits": logits,
        "classification_labels": labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_classification_loss")(ret["classification_loss"])
    acc = getattr(pl_module, f"{phase}_classification_accuracy")(
        ret["classification_logits"], ret["classification_labels"]
    )

    pl_module.log(f"classification/{phase}/loss", loss)
    pl_module.log(f"classification/{phase}/accuracy", acc)

    return ret


def compute_mpp(pl_module, batch):
    infer = pl_module.infer(batch, mask_grid=True)

    mpp_logits = pl_module.mpp_head(infer["grid_feats"])  # [B, max_image_len+2, bins]
    mpp_logits = mpp_logits[:, :-1, :]  # ignore volume embedding, [B, max_image_len+1, bins]
    mpp_labels = infer["grid_labels"]  # [B, max_image_len+1, C=1]

    mask = mpp_labels != -100.  # [B, max_image_len, 1]

    # masking
    mpp_logits = mpp_logits[mask.squeeze(-1)]  # [mask, bins]
    mpp_labels = mpp_labels[mask].long()  # [mask]

    mpp_loss = F.cross_entropy(mpp_logits, mpp_labels)

    ret = {
        "mpp_loss": mpp_loss,
        "mpp_logits": mpp_logits,
        "mpp_labels": mpp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(
        ret["mpp_logits"], ret["mpp_labels"]
    )

    pl_module.log(f"mpp/{phase}/loss", loss)
    pl_module.log(f"mpp/{phase}/accuracy", acc)

    return ret


def compute_mtp(pl_module, batch):
    infer = pl_module.infer(batch, mask_grid=False)
    mtp_logits = pl_module.mtp_head(infer["cls_feats"])  # [B, hid_dim]
    mtp_labels = torch.LongTensor(batch["mtp"]).to(mtp_logits.device)  # [B]

    mtp_loss = F.cross_entropy(mtp_logits, mtp_labels)  # [B]

    ret = {
        "mtp_loss": mtp_loss,
        "mtp_logits": mtp_logits,
        "mtp_labels": mtp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mtp_loss")(ret["mtp_loss"])
    acc = getattr(pl_module, f"{phase}_mtp_accuracy")(
        ret["mtp_logits"], ret["mtp_labels"]
    )

    pl_module.log(f"mtp/{phase}/loss", loss)
    pl_module.log(f"mtp/{phase}/accuracy", acc)

    return ret


def compute_vfp(pl_module, batch):
    infer = pl_module.infer(batch, mask_grid=False)

    vfp_logits = pl_module.vfp_head(infer["cls_feats"]).squeeze(-1)  # [B]
    vfp_labels = torch.FloatTensor(batch["vfp"]).to(vfp_logits.device)

    assert len(vfp_labels.shape) == 1

    vfp_loss = F.mse_loss(vfp_logits, vfp_labels)
    ret = {
        "vfp_loss": vfp_loss,
        "vfp_logits": vfp_logits,
        "vfp_labels": vfp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vfp_loss")(ret["vfp_loss"])
    mae = getattr(pl_module, f"{phase}_vfp_mae")(
        mean_absolute_error(ret["vfp_logits"], ret["vfp_labels"])
    )

    pl_module.log(f"vfp/{phase}/loss", loss)
    pl_module.log(f"vfp/{phase}/mae", mae)

    return ret


def compute_ggm(pl_module, batch):
    pos_len = len(batch["grid"]) // 2
    neg_len = len(batch["grid"]) - pos_len
    ggm_labels = torch.cat(
        [torch.ones(pos_len), torch.zeros(neg_len)]
    ).to(pl_module.device)

    ggm_images = []
    for i, (bti, bfi) in enumerate(zip(batch["grid"], batch["false_grid"])):
        if ggm_labels[i] == 1:
            ggm_images.append(bti)
        else:
            ggm_images.append(bfi)

    ggm_images = torch.stack(ggm_images, dim=0)

    batch = {k: v for k, v in batch.items()}
    batch["grid"] = ggm_images

    infer = pl_module.infer(batch, mask_grid=False)
    ggm_logits = pl_module.ggm_head(infer["cls_feats"])  # cls_feats
    ggm_loss = F.cross_entropy(ggm_logits, ggm_labels.long())

    ret = {
        "ggm_loss": ggm_loss,
        "ggm_logits": ggm_logits,
        "ggm_labels": ggm_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_ggm_loss")(ret["ggm_loss"])
    acc = getattr(pl_module, f"{phase}_ggm_accuracy")(
        ret["ggm_logits"], ret["ggm_labels"]
    )

    pl_module.log(f"ggm/{phase}/loss", loss)
    pl_module.log(f"ggm/{phase}/accuracy", acc)

    return ret


def compute_moc(pl_module, batch):
    infer = pl_module.infer(batch, mask_grid=False)
    moc_logits = pl_module.moc_head(
        infer["graph_feats"][:, 1:, :]).flatten()  # [B, max_graph_len] -> [B * max_graph_len]
    moc_labels = infer["mo_labels"].to(moc_logits).flatten()  # [B, max_graph_len] -> [B * max_graph_len]
    mask = moc_labels != -100

    moc_loss = F.binary_cross_entropy_with_logits(input=moc_logits[mask],
                                                  target=moc_labels[mask])  # [B * max_graph_len]

    ret = {
        "moc_loss": moc_loss,
        "moc_logits": moc_logits,
        "moc_labels": moc_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_moc_loss")(ret["moc_loss"])
    acc = getattr(pl_module, f"{phase}_moc_accuracy")(
        nn.Sigmoid()(ret["moc_logits"]), ret["moc_labels"].long()
    )

    pl_module.log(f"moc/{phase}/loss", loss)
    pl_module.log(f"moc/{phase}/accuracy", acc)

    return ret


def compute_bbp(pl_module, batch):
    infer = pl_module.infer(batch, mask_grid=False)
    bbp_logits = pl_module.bbp_head(infer["cls_feats"])  # [B, num_bbs]
    # labels
    batch_size, num_bbs = bbp_logits.shape
    bbp_labels = torch.zeros([batch_size, num_bbs]).to(bbp_logits.device)  # [B, num_bbs]
    for bi, bb in enumerate(batch["bbp"]):
        for b in bb:
            bbp_labels[bi][b] = 1
    # flatten
    bbp_logits = bbp_logits.flatten()
    bbp_labels = bbp_labels.flatten()

    bbp_loss = F.binary_cross_entropy_with_logits(input=bbp_logits,
                                                  target=bbp_labels)

    ret = {
        "bbp_loss": bbp_loss,
        "bbp_logits": bbp_logits,
        "bbp_labels": bbp_labels,
    }

    # call update() loss and acc
    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_bbp_loss")(ret["bbp_loss"])
    acc = getattr(pl_module, f"{phase}_bbp_accuracy")(
        nn.Sigmoid()(ret["bbp_logits"]), ret["bbp_labels"].long()
    )

    pl_module.log(f"bbp/{phase}/loss", loss)
    pl_module.log(f"bbp/{phase}/accuracy", acc)

    return ret
