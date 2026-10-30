def evolved_ring_kv(kv_chunks, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    num_nodes = max(1, world_size // 32)
    rpn = world_size // num_nodes
    intra = [[n * rpn + i for i in range(rpn)] for n in range(num_nodes)]
    inter = [[n * rpn + i for n in range(num_nodes)] for i in range(rpn)]
    outs = []
    for c in kv_chunks:
        g1 = xm.all_gather(c.unsqueeze(0), dim=0, groups=intra)
        if num_nodes == 1:
            outs.append(g1.reshape(-1)); continue
        g2 = xm.all_gather(g1.unsqueeze(0), dim=0, groups=inter)
        outs.append(g2.reshape(-1))
    return outs
