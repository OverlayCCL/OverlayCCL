def evolved_ring_kv(kv_chunks, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    outs = []
    for c in kv_chunks:
        buf = torch.zeros(world_size, c.numel(), dtype=c.dtype, device=c.device)
        buf[rank] = c
        outs.append(xm.all_reduce(xm.REDUCE_SUM, buf).reshape(-1))
    return outs
