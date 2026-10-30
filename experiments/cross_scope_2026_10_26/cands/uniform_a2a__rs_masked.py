def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    x = input_tensor.view(world_size, chunk_size)
    buf = torch.zeros(world_size, world_size, chunk_size, dtype=input_tensor.dtype, device=input_tensor.device)
    buf[:, rank, :] = x
    out = xm.reduce_scatter(xm.REDUCE_SUM, buf, scale=1.0, scatter_dim=0, shard_count=world_size)
    return out.reshape(-1)
