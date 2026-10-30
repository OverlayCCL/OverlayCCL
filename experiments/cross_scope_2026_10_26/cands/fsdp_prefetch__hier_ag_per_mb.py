def evolved_fsdp_prefetch(shards, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    num_nodes = max(1, world_size // 32)
    rpn = world_size // num_nodes
    intra = [[nn * rpn + i for i in range(rpn)] for nn in range(num_nodes)]
    inter = [[nn * rpn + i for nn in range(num_nodes)] for i in range(rpn)]
    outs = []
    for m in range(M):
        st = torch.stack(shards[m], dim=0)
        g = xm.all_gather(st, dim=-1, groups=intra)
        if num_nodes > 1:
            g = xm.all_gather(g, dim=-1, groups=inter)
        outs.append([g[L] for L in range(N_LAYERS)])
    return outs
