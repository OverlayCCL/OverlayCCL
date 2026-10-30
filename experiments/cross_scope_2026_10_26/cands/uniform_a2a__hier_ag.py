def evolved_uniform_a2a(input_tensor, chunk_size, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    num_nodes = max(1, world_size // 32)
    rpn = world_size // num_nodes
    node, local = rank // rpn, rank % rpn
    intra = [[n * rpn + i for i in range(rpn)] for n in range(num_nodes)]
    g1 = xm.all_gather(input_tensor.unsqueeze(0), dim=0, groups=intra)
    if num_nodes == 1:
        return g1[:, rank * chunk_size:(rank + 1) * chunk_size].reshape(-1)
    inter = [[n * rpn + i for n in range(num_nodes)] for i in range(rpn)]
    cols = torch.stack([g1[:, (n * rpn + local) * chunk_size:(n * rpn + local + 1) * chunk_size] for n in range(num_nodes)], dim=0)
    g2 = xm.all_gather(cols.unsqueeze(0), dim=0, groups=inter)
    return g2[:, node, :, :].reshape(-1)
