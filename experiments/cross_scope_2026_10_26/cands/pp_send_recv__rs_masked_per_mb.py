def evolved_pp_send_recv(activations, src_stage, half, M, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    stage = 0 if rank < half else 1
    outs = []
    for m in range(M):
        a = activations[m]
        buf = torch.zeros(world_size, *a.shape, dtype=a.dtype, device=a.device)
        if stage == src_stage:
            buf[rank] = a
            buf[rank + half if src_stage == 0 else rank - half] = a
        r = xm.reduce_scatter(xm.REDUCE_SUM, buf, scale=1.0, scatter_dim=0, shard_count=world_size)
        outs.append(r[0])
    return outs
