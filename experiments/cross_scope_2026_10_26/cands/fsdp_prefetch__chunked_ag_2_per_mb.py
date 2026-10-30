def evolved_fsdp_prefetch(shards, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    outs = []
    for m in range(M):
        st = torch.stack(shards[m], dim=0)
        h = st.shape[1] // 2
        g0 = xm.all_gather(st[:, :h, :].contiguous(), dim=-1)
        g1 = xm.all_gather(st[:, h:, :].contiguous(), dim=-1)
        g = torch.cat([g0, g1], dim=1)
        outs.append([g[L] for L in range(N_LAYERS)])
    return outs
