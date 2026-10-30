def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    num_nodes = max(1, world_size // 32)
    rpn = world_size // num_nodes
    intra = [[nn * rpn + i for i in range(rpn)] for nn in range(num_nodes)]
    inter = [[nn * rpn + i for nn in range(num_nodes)] for i in range(rpn)]
    shapes = [partials[m][L].shape for m in range(M) for L in range(N_LAYERS)]
    flat = torch.cat([partials[m][L].flatten() for m in range(M) for L in range(N_LAYERS)], dim=0)
    r = xm.all_reduce(xm.REDUCE_SUM, flat, groups=intra)
    if num_nodes > 1:
        r = xm.all_reduce(xm.REDUCE_SUM, r, groups=inter)
    outs = []; off = 0
    for m in range(M):
        row = []
        for L in range(N_LAYERS):
            n = 1
            for d in shapes[m * N_LAYERS + L]: n *= d
            row.append(r[off:off + n].reshape(shapes[m * N_LAYERS + L])); off += n
        outs.append(row)
    return outs
