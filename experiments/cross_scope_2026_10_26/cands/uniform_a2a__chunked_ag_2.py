def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    K = 2
    part = chunk_size // K
    x = input_tensor.view(world_size, chunk_size)
    outs = []
    for k in range(K):
        g = xm.all_gather(x[:, k * part:(k + 1) * part].contiguous().unsqueeze(0), dim=0)
        outs.append(g[:, rank, :])
    return torch.cat(outs, dim=1).reshape(-1)
