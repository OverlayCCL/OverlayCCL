def evolved_pp_send_recv(activations, src_stage, half, M, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    stage = 0 if rank < half else 1
    src = rank if stage == src_stage else (rank - half if src_stage == 0 else rank + half)
    st = torch.stack(activations, dim=0)
    g = xm.all_gather(st.unsqueeze(0), dim=0)
    return [g[src, m] for m in range(M)]
