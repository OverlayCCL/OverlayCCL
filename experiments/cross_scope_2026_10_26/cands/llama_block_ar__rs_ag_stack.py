def evolved_llama_block(attn_partial, mlp_partial, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    st = torch.stack([attn_partial, mlp_partial], dim=0)
    flat = st.reshape(-1)
    n = flat.numel(); pad = (-n) % world_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)], dim=0)
    rs = xm.reduce_scatter(xm.REDUCE_SUM, flat, scale=1.0, scatter_dim=0, shard_count=world_size)
    r = xm.all_gather(rs, dim=0)[:n].reshape(st.shape)
    return [r[0], r[1]]
