def evolved_llama_block(attn_partial, mlp_partial, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    num_nodes = max(1, world_size // 32)
    rpn = world_size // num_nodes
    intra = [[nn * rpn + i for i in range(rpn)] for nn in range(num_nodes)]
    inter = [[nn * rpn + i for nn in range(num_nodes)] for i in range(rpn)]
    st = torch.stack([attn_partial, mlp_partial], dim=0)
    r = xm.all_reduce(xm.REDUCE_SUM, st, groups=intra)
    if num_nodes > 1:
        r = xm.all_reduce(xm.REDUCE_SUM, r, groups=inter)
    return [r[0], r[1]]
