import torch
import numpy as np
import logging
import math


def load_ckpt(ckpt_path:str):
    print("load checkpoint ...")
    return torch.load(ckpt_path, map_location="cuda")


def efficient_rank(singular_values):
    # use exp(-\sum_i p_i*log(p_i)) to compute efficient rank
    p = singular_values / singular_values.sum()
    H = -(p*torch.log(p)).sum()  # entropy
    eff_rank = torch.exp(H)
    return eff_rank

@torch.no_grad()
def ckpt_sim(ckpt1, ckpt2, rank=16):
    pwconv1_state = {}
    pwconv2_state = {}
    gamma_state = {}
    bias_state = {}
    norm_state = {}
    downsample_state = {}
    if "state_dict" in ckpt1:
        ckpt1 = ckpt1["state_dict"]
    if "state_dict" in ckpt2:
        ckpt2 = ckpt2["state_dict"]

    if "model" in ckpt1:
        ckpt1 = ckpt1["model"]
    if "model" in ckpt2:
        ckpt2 = ckpt2["model"]
    for name, param in ckpt1.items():
        # print("Parse param: {}".format(name))
        # if "lora" in name:
        #     print(name, param.norm())
        # if "gamma" in name:
        #     # p2 = ckpt2[name.split("backbone.")[-1]]
        #     p2 = ckpt2[name]
        #     gamma_state[name] = [param, p2]
        # if "bias" in name and "backbone" in name and not "norm" in name:
        #     p2 = ckpt2[name.split("backbone.")[-1]]
        #     bias_state[name] = [param, p2]
        # if "norm" in name and "backbone" in name:
        #     norm_name = name.split("backbone.")[-1]
        #     if norm_name in ckpt2:
        #         norm_state[name] = [param, ckpt2[norm_name]]
        if "pwconv1" in name and "weight" in name:
            # p2 = ckpt2[name.split("backbone.")[-1]]
            p2 = ckpt2[name]
            u1, s1, v1 = torch.svd(param-p2)
            u2, s2, v2 = torch.svd(torch.cat([param-p2, param], dim=-1)) # [C, r]
            # print(param.shape, p2.shape)

            # check whether these base vectors are in the same subspace
            if "pwconv1" in name:
                # u1 = u1[:, :32]
                # u2 = u2[:, :32]
                c = u1.shape[-1]
                u = torch.cat([u1], dim=-1) # [u1, u2]
                # rank = torch.linalg.matrix_rank(u)
                rank = efficient_rank(s1)
                print(rank, c)
                # fullrank = torch.linalg.matrix_rank(torch.cat([u1, u2], dim=-1))
                fullrank = efficient_rank(s2)
                s1_mean, s1_std = s1.mean(), s1.std()
                s2_mean, s2_std = s2.mean(), s2.std()
                u_cos = torch.matmul(u1.transpose(0, 1), u2).diag().mean()
                pwconv1_state[name] = [rank.cpu(), s1.shape[0], s1_mean.cpu(), s1_std.cpu(), s2_mean.cpu(), s2_std.cpu(), u_cos.cpu(), fullrank.cpu()]

                # name_pwconv2 = name.replace("pwconv1", "pwconv2")
                # p = ckpt1[name_pwconv2]
                # pp = ckpt2[name_pwconv2.split("backbone.")[-1]]
                # u, s, v = torch.svd(pp - p)
                # uv = torch.cat([u1, v], dim=-1)
                # rank_cross = torch.linalg.matrix_rank(uv)
                # # pwconv1_state[name][0] = rank_cross.cpu()
                # pwconv1_state[name][4] = s.mean().cpu()
            # else:
            #     v1 = v1[:, :32]
            #     v2 = v2[:, :32]
            #     s1 = s1[:32]
            #     v = torch.cat([v1, v2], dim=-1)
            #     rank = torch.linalg.matrix_rank(v)
            #     v_cos = torch.matmul(v1.transpose(0, 1), v2).mean()
            #     pwconv2_state[name] = [rank.cpu(), s1.shape[0], s1_mean.cpu(), s1_std.cpu(), s2_mean.cpu(), s2_std.cpu(), u_cos.cpu()]
    return pwconv1_state, pwconv2_state, gamma_state, bias_state, norm_state


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # ckpt_p1 = "logs/convnext_s_correlated_shareA_indep_redu10_prune/iter_48000.pth"
    # ckpt_p1 = "../coco_detection/logs/convnext_b_fulltune/epoch_1.pth"
    # ckpt_p2 = "../coco_detection/logs/convnext_b_fulltune/epoch_2.pth"
    # ckpt_p2 = "pretrain/convnext_base_22k_224.pth"
    ckpt_p1 = "logs/convnext_s_fulltune/iter_160000.pth"
    ckpt_p2 = "logs/convnext_s_fulltune/iter_144000.pth"
    # ckpt_p2 = "logs/convnext_b_useloradwconv_usefilter_pw12correlated_redu16/iter_16000.pth"

    ckpt1, ckpt2 = load_ckpt(ckpt_p1), load_ckpt(ckpt_p2)
    pwconv1_state, pwconv2_state, gamma_state, bias_state, norm_state = ckpt_sim(ckpt1, ckpt2)
    state = pwconv1_state.values()
    for name, param in gamma_state.items():
        diff = param[0] - param[1]
        print(name, diff.min().item(), param[0][diff.argmin()].item(), param[1][diff.argmin()].item(), diff.max().item(), param[0][diff.argmax()].item(), param[1][diff.argmax()].item())
    # plot rank, and delta rank
    plt.figure(figsize=(8, 2))
    rank = [item[0] for item in state]
    C = [item[1] for item in state]
    fullrank = [item[-1] for item in state]
    plt.plot(np.arange(len(rank)), rank, 'r-o', label='Rank of delta w')
    plt.plot(np.arange(len(C)), C, 'g-*', label='Rank of w')
    plt.plot(np.arange(len(fullrank)), fullrank, 'y-^', label='Rank of [w, delta w]')
    plt.xlabel("layer index")
    plt.legend()
    plt.grid()
    plt.savefig("figures/convnext_b_eff_rank_dimension_pretrain_fulltune_160k.png", dpi=150, bbox_inches='tight')
    plt.close()

    # plot mean std
    s1_mean = [item[2] for item in state]
    s1_std = [item[3]*0 for item in state]
    s2_mean = [item[4] for item in state]
    s2_std = [item[5]*0 for item in state]
    plt.figure(figsize=(8, 2.6))
    plt.errorbar(np.arange(len(s1_mean)), s1_mean, yerr=s1_std, fmt='o-', ecolor='g', color='g', label='full tuning')
    plt.errorbar(np.arange(len(s2_mean)), s2_mean, yerr=s2_std, fmt='o-', ecolor='purple', color='purple', label='full tuning')
    plt.grid()
    plt.legend()
    plt.savefig("figures/convnext_b_mean_std_fulltune_128k_160k.png", dpi=150)
    plt.close()

    # plot u sim
    cos_sim = [item[-1] for item in state]
    plt.figure(figsize=(8, 2.6))
    plt.plot(np.arange(len(cos_sim)), cos_sim, 'g-*', label='similarity between full-peft weight bases')
    plt.grid()
    plt.savefig("figures/convnext_b_cos_sim_fulltune_128k_160k.png", dpi=150)
    plt.close()




