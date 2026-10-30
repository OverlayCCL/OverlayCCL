def evolved_dxe(logits_local, targets, V_local, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    num_nodes = max(1, world_size // 32)
    rpn = world_size // num_nodes
    intra = [[n * rpn + i for i in range(rpn)] for n in range(num_nodes)]
    inter = [[n * rpn + i for n in range(num_nodes)] for i in range(rpn)]
    lo, hi = rank * V_local, (rank + 1) * V_local
    local_max = logits_local.max(dim=-1).values
    gmax = xm.all_reduce(xm.REDUCE_MAX, local_max, groups=intra)
    if num_nodes > 1:
        gmax = xm.all_reduce(xm.REDUCE_MAX, gmax, groups=inter)
    sumexp = (logits_local - gmax.unsqueeze(-1)).exp().sum(dim=-1)
    in_shard = (targets >= lo) & (targets < hi)
    t = logits_local.gather(1, (targets - lo).clamp(0, V_local - 1).unsqueeze(1)).squeeze(1)
    t = torch.where(in_shard, t, torch.zeros_like(t))
    pack = torch.stack([sumexp, t], dim=1)
    g = xm.all_reduce(xm.REDUCE_SUM, pack, groups=intra)
    if num_nodes > 1:
        g = xm.all_reduce(xm.REDUCE_SUM, g, groups=inter)
    return (g[:, 0].log() + gmax - g[:, 1]).mean()
