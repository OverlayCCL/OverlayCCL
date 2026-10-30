def evolved_pp_send_recv(activations, src_stage, half, M, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    stage = 0 if rank < half else 1
    a0 = activations[0]
    buf = torch.zeros(world_size, M, *a0.shape, dtype=a0.dtype, device=a0.device)
    if stage == src_stage:
        st = torch.stack(activations, dim=0)
        buf[rank] = st
        buf[rank + half if src_stage == 0 else rank - half] = st
    r = xm.reduce_scatter(xm.REDUCE_SUM, buf, scale=1.0, scatter_dim=0, shard_count=world_size)
    return [r[0, m] for m in range(M)]
