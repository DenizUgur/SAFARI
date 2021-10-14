import os
import torch
import importlib
import numpy as np
from tqdm import tqdm
from glob import glob

import matplotlib.pyplot as plt

from utils.option import args
from data import create_loader
from metric import metric as module_metric


MODELS = glob(os.path.join(args.pre_train, "*.pt"))


def graph():
    data = np.load(os.path.join(args.pre_train, "summary.npy"), allow_pickle=True)

    names = []
    datas = []

    for model, model_data in zip(MODELS, data):
        name = int(model.split("256/G")[1].split(".")[0])
        block = []
        for entry in model_data:
            block.append([*entry.values()])
        block = np.array(block)

        names.append(name)
        datas.append(block)
    names = np.array(names)
    datas = np.array(datas)
    datas = np.swapaxes(datas, 2, 1)
    datas = np.mean(datas, axis=2)

    datas[:, 0] *= 1.0 / datas[:, 0].max()
    datas[:, 1] *= 1.0 / datas[:, 1].max()
    datas[:, 2] *= 1.0 / datas[:, 2].max()

    _indsorted = names.argsort()
    datas = datas[_indsorted[::-1]]

    m1, m2, m3 = 1, 1, 1
    quality = (m1 * (1 - datas[:, 0]) + m2 * datas[:, 1] + m3 * datas[:, 2]) / 3

    fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharex=True)

    axs[0, 0].plot(datas[:, 0], c="red")
    axs[0, 1].plot(datas[:, 1], c="red")
    axs[1, 0].plot(datas[:, 2], c="red")
    axs[1, 1].plot(quality, c="green")

    fig.tight_layout()
    plt.savefig(os.path.join(args.pre_train, "summary.png"))
    plt.show()


def main():
    args.world_size = 1

    data = []

    for model_name in tqdm(MODELS, position=1):
        data_loader = create_loader(args)
        name = model_name.split("pconv256/")[1].split(".")[0]

        # Model and version
        net = importlib.import_module("model." + args.model)
        model = net.InpaintGenerator(args).cuda()
        model.load_state_dict(torch.load(model_name))
        model.eval()

        results = []
        for i in tqdm(range(args.iterations), position=2):
            target, mask, _, _ = next(data_loader)
            target = target.cuda()
            mask = mask.cuda()

            with torch.no_grad():
                masked_tensor = (target * (1 - mask).float()) + mask
                pred_tensor = model(masked_tensor, mask)
                comp_tensor = pred_tensor * mask + target * (1 - mask)

                # metrics prepare for image assesments
                metrics = {
                    met: getattr(module_metric, met) for met in ["mae", "psnr", "ssim"]
                }
                evaluation_scores = {key: 0 for key, val in metrics.items()}
                for key, val in metrics.items():
                    evaluation_scores[key] = val(
                        target.cpu().numpy().reshape((256, 256)),
                        comp_tensor.cpu().numpy().reshape((256, 256)),
                        num_worker=1,
                        no_tqdm=True,
                    )
                results.append(evaluation_scores)
        data.append(results)

    data = np.array(data)
    np.save(os.path.join(args.pre_train, "summary"), data)
    print(data)


if __name__ == "__main__":
    graph()