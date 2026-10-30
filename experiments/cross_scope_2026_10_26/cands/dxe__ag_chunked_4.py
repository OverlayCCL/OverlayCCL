def evolved_dxe(logits_local, targets, V_local, rank, world_size, num_devices, cores_per_device, xm, torch, num_nodes=1):
    K = 4
    part = V_local // K
    r = targets // V_local
    m = None; s = None; tl = None
    for k in range(K):
        g = xm.all_gather(logits_local[:, k * part:(k + 1) * part].contiguous(), dim=1)
        cm = g.max(dim=-1).values
        off = targets - r * V_local - k * part
        inside = (off >= 0) & (off < part)
        col = r * part + off.clamp(0, part - 1)
        tk = g.gather(1, col.unsqueeze(1)).squeeze(1)
        if m is None:
            m = cm
            s = (g - m.unsqueeze(-1)).exp().sum(dim=-1)
            tl = torch.where(inside, tk, torch.zeros_like(tk))
        else:
            nm = torch.where(m > cm, m, cm)
            s = s * (m - nm).exp() + (g - nm.unsqueeze(-1)).exp().sum(dim=-1)
            m = nm
            tl = torch.where(inside, tk, tl)
    return (s.log() + m - tl).mean()
