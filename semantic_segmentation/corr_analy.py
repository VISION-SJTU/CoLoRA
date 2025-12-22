import torch
import os
import numpy as np


def load_ckpt(ckpt_path:str):
    print("load checkpoint ...")
    return torch.load(ckpt_path, map_location="cuda")

@torch.no_grad()
def ckpt_sim(ckpt1, ckpt2):
    pwconv_states = {}
    if "state_dict" in ckpt1:
        ckpt1 = ckpt1["state_dict"]
    if "state_dict" in ckpt2:
        ckpt2 = ckpt2["state_dict"]
    if "model" in ckpt1:
        ckpt1 = ckpt1["model"]
    if "model" in ckpt2:
        ckpt2 = ckpt2["model"]
    for name, param in ckpt1.items():
        if "pwconv" in name and "weight" in name:
            p2 = ckpt2[name]
            u1, s1, v1 = torch.svd(p2 - param)
            # check whether these base vectors are in the same subspace
            if "pwconv1" in name:
                name_pwconv2 = name.replace("pwconv1", "pwconv2")
                p = ckpt1[name_pwconv2]
                pp = ckpt2[name_pwconv2]
                u2, s2, v2 = torch.svd(pp - p)
            pwconv_states[name] = [s1.cpu(), s2.cpu()]
    del ckpt1, ckpt2
    return pwconv_states

@torch.no_grad()
def read_all(ckpt_folder):
    start_iter = 16000
    interval = 16000
    ckpt_format = "iter_{}.pth"
    for i in range(9):
        ckpt1 = load_ckpt(os.path.join(ckpt_folder, ckpt_format.format(start_iter)))
        ckpt2 = load_ckpt(os.path.join(ckpt_folder, ckpt_format.format(start_iter+interval)))
        states = ckpt_sim(ckpt1, ckpt2)
        torch.save(states, "state_{}.pth".format(start_iter))
        start_iter += interval

@torch.no_grad()
def read_states():
    start_iter = 16000
    interval = 16000
    states = dict()
    for i in range(9):
        state = torch.load("state_{}.pth".format(start_iter))
        for name, item in state.items():
            if name not in states:
                states[name] = [item]
            else:
                states[name].append(item)
        start_iter = start_iter + interval
    return states


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # ckpt_folder = "logs/convnext_b_fulltune"
    # read_all(ckpt_folder)
    states = read_states()
    # plt.figure(figsize=(32, 3))
    keys = list(states.keys())
    keys = [k for k in keys if "pwconv1" in k]
    print(len(keys))
    plt.figure(figsize=(24, 24))
    for i in range(6):
        for j in range(6):
            idx = 6*i + j
            plt.subplot(6, 6, idx+1)
            k = keys[idx]
            plt.plot(np.arange(1, 10), [states[k][i][0].mean() for i in range(9)], "r-o", label=k.split("backbone.")[-1].split(".weight")[0])
            plt.plot(np.arange(1, 10), [states[k][i][1].mean() for i in range(9)], "g-*", label=k.split("backbone.")[-1].replace("pwconv1", "pwconv2").split(".weight")[0])
            plt.xlabel("Interval (16k iters)")
            plt.grid()
            plt.legend()
    plt.savefig("figures/correlate_stages_all.png", dpi=150, bbox_inches='tight')

    # k = keys[0]
    # plt.subplot(1, 4, 1)
    # plt.plot(np.arange(1, 10), [states[k][i][0].mean() for i in range(9)], "r-o", label=k.split("backbone.")[-1].split(".weight")[0])
    # plt.plot(np.arange(1, 10), [states[k][i][1].mean() for i in range(9)], "g-*", label=k.split("backbone.")[-1].replace("pwconv1", "pwconv2").split(".weight")[0])
    # plt.xlabel("Interval (16k iters)")
    # plt.grid()
    # plt.legend()
    # k = keys[3]
    # plt.subplot(1, 4, 2)
    # plt.plot(np.arange(1, 10), [states[k][i][0].mean() for i in range(9)], "r-o", label=k.split("backbone.")[-1].split(".weight")[0])
    # plt.plot(np.arange(1, 10), [states[k][i][1].mean() for i in range(9)], "g-*", label=k.split("backbone.")[-1].replace("pwconv1", "pwconv2").split(".weight")[0])
    # plt.xlabel("Interval (16k iters)")
    # plt.grid()
    # plt.legend()
    # k = keys[6]
    # plt.subplot(1, 4, 3)
    # plt.plot(np.arange(1, 10), [states[k][i][0].mean() for i in range(9)], "r-o", label=k.split("backbone.")[-1].split(".weight")[0])
    # plt.plot(np.arange(1, 10), [states[k][i][1].mean() for i in range(9)], "g-*", label=k.split("backbone.")[-1].replace("pwconv1", "pwconv2").split(".weight")[0])
    # plt.xlabel("Interval (16k iters)")
    # plt.grid()
    # plt.legend()
    # k = keys[33]
    # plt.subplot(1, 4, 4)
    # plt.plot(np.arange(1,10), [states[k][i][0].mean() for i in range(9)], "r-o", label=k.split("backbone.")[-1].split(".weight")[0])
    # plt.plot(np.arange(1,10), [states[k][i][1].mean() for i in range(9)], "g-*", label=k.split("backbone.")[-1].replace("pwconv1", "pwconv2").split(".weight")[0])
    # plt.grid()
    # plt.xlabel("Interval (16k iters)")
    # plt.legend()
    # plt.savefig("figures/correlate_stages.png", dpi=150, bbox_inches='tight')
    


