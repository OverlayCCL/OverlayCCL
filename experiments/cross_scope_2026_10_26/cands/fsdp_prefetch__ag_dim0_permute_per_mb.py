def evolved_fsdp_prefetch(shards, M, N_LAYERS, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    outs = []
    for m in range(M):
        st = torch.stack(shards[m], dim=0)
        L, DM, S = st.shape
        g = xm.all_gather(st.unsqueeze(0), dim=0)
        g = g.permute(1, 2, 0, 3).reshape(L, DM, world_size * S)
        outs.append([g[L_] for L_ in range(N_LAYERS)])
    return outs
