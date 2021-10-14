import importlib
import numpy as np
import matplotlib.pyplot as plt

import torch
from math import ceil

from utils.option import args
from metric import metric as module_metric
from data.dataset import TerrainDataset

from timeit import default_timer as dt


def demo(args):
    args.world_size = 1
    IS = args.image_size
    dataset = TerrainDataset(
        args.dir_train,
        dataset_type="train",
        randomize=True,
        block_variance=1,
        random_state=42,
        block_dimension=args.block_dimension,
        observer_height=0.75,
        patch_size=ceil(args.image_size / 100) * 10,
        sample_size=args.image_size,
        observer_pad=args.image_size // 4,
    )

    # Model and version
    net = importlib.import_module("model." + args.model)
    model = net.InpaintGenerator(args)
    model.load_state_dict(torch.load(args.pre_train))
    model.eval()
    model.cuda()

    for images, masks in dataset:
        images = images.unsqueeze(0)
        masks = masks.unsqueeze(0)

        images, masks = images.cuda(), masks.cuda()
        images_masked = (images * (1 - masks).float()) + masks
        print(f"[**] inpainting ... ")

        with torch.no_grad():
            start = dt()
            pred_tensor = model(images_masked, masks)
            comp_tensor = (1 - masks) * images + masks * pred_tensor
            print(f"Took {(dt() - start):.3f} seconds")

            # metrics prepare for image assesments
            metrics = {
                met: getattr(module_metric, met) for met in ["mae", "psnr", "ssim"]
            }
            evaluation_scores = {key: [] for key, val in metrics.items()}
            for i in range(args.block_dimension):
                for (key, val) in metrics.items():
                    evaluation_scores[key].append(
                        val(
                            images[0, i].cpu().numpy().reshape((IS, IS)),
                            comp_tensor[0, i].cpu().numpy().reshape((IS, IS)),
                            num_worker=1,
                        )
                        * (100 if key == "ssim" else 1)
                    )

            plt.style.use("dark_background")
            fig, axs = plt.subplots(
                4, args.block_dimension, figsize=(32, 20), sharex=True, sharey=True
            )

            new_size = (args.block_dimension, IS, IS)
            images = images.reshape(new_size)
            masks = masks.reshape(new_size)
            masked = np.copy(images.cpu().numpy())
            masked[masks.cpu().numpy() == 1] = np.nan
            pred_tensor = pred_tensor.reshape(new_size)
            comp_tensor = comp_tensor.reshape(new_size)

            images = images.cpu().numpy()
            levels = np.arange(
                images.min(), images.max(), (images.max() - images.min()) / 200
            )
            cmap = "terrain"

            for i in range(args.block_dimension):
                axs[0, i].contourf(images[i], levels=levels, cmap=cmap)
                axs[1, i].contourf(masked[i], levels=levels, cmap=cmap)
                axs[2, i].contourf(pred_tensor[i].cpu().numpy(), levels=levels, cmap=cmap)
                axs[3, i].contourf(comp_tensor[i].cpu().numpy(), levels=levels, cmap=cmap)

                axs[0, i].set_title(
                    " ".join(
                        [
                            "{}: {:.2f}".format(key[0], val[i])
                            for key, val in evaluation_scores.items()
                        ],
                    ),
                    fontsize=16,
                )

            fig.tight_layout(rect=[0, 0.03, 1, 0.92])
            fig.suptitle("AOT-GAN Results", fontsize=32)
            plt.show()


if __name__ == "__main__":
    demo(args)
