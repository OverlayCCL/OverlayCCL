def evolved_tp_mlp(partials, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    shapes = [partials[m][L].shape for m in range(M) for L in range(N_LAYERS)]
    flat = torch.cat([partials[m][L].flatten() for m in range(M) for L in range(N_LAYERS)], dim=0)
    n = flat.numel(); pad = (-n) % world_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)], dim=0)
    rs = xm.reduce_scatter(xm.REDUCE_SUM, flat, scale=1.0, scatter_dim=0, shard_count=world_size)
    r = xm.all_gather(rs, dim=0)
    outs = []; off = 0
    for m in range(M):
        row = []
        for L in range(N_LAYERS):
            n = 1
            for d in shapes[m * N_LAYERS + L]: n *= d
            row.append(r[off:off + n].reshape(shapes[m * N_LAYERS + L])); off += n
        outs.append(row)
    return outs
