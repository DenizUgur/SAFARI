from .dataset import TerrainDataset
from math import ceil

from torch.utils.data import DataLoader


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def create_loader(args):
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
    print(f"Dataset range is {dataset.data_range}")
    print(f"Dataset size is {len(dataset)}")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.world_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return sample_data(data_loader)