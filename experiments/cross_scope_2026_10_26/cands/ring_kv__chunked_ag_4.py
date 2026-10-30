def evolved_ring_kv(kv_chunks, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    K = 4
    outs = []
    for c in kv_chunks:
        part = c.numel() // K
        gs = [xm.all_gather(c[k * part:(k + 1) * part].unsqueeze(0), dim=0) for k in range(K)]
        outs.append(torch.cat(gs, dim=1).reshape(-1))
    return outs
