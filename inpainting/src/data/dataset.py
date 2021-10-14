import os
import math
import torch
import random
import pickle
import hashlib
import rasterio
import numpy as np
from glob import glob
from tqdm import tqdm
from scipy import stats
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from skimage.draw import rectangle_perimeter, line


class Viewshed:
    def __init__(
        self,
        observer_height=0.75,
        observer_pad=50,
        observer_steps=4,
        observer_view_angle=120,
        data_range=None,
    ):
        self.observer_height = observer_height / data_range
        self.observer_pad = observer_pad
        self.observer_steps = observer_steps
        self.observer_view_angle = observer_view_angle

    def get_poi(self, dem):
        h, w = dem.shape
        center = dem[
            self.observer_pad - self.observer_pad // 2 : h - self.observer_pad // 2,
            self.observer_pad - self.observer_pad // 2 : w - self.observer_pad // 2,
        ]

        # * Find maximum point and furthest point
        max = np.unravel_index(np.argmax(center, axis=None), center.shape)
        pad = 0
        furthest = (
            h - pad if max[0] < h // 2 else pad,
            w - pad if max[1] < w // 2 else pad,
        )

        # * Draw a line and subdivide it
        ys, xs = line(*max, *furthest)
        comb = np.hstack((ys.reshape(-1, 1), xs.reshape(-1, 1)))[: h // 2]
        steps = np.linspace(
            0, len(comb), num=self.observer_steps, endpoint=False, dtype=int
        )
        steps = np.take(comb, steps, axis=0)

        # * Find the starting point and movement angle
        dist1 = ((steps[0, 0] - h // 2) ** 2 + (steps[0, 1] - w // 2) ** 2) ** 0.5
        dist2 = ((steps[-1, 0] - h // 2) ** 2 + (steps[-1, 1] - w // 2) ** 2) ** 0.5

        if dist1 < dist2:
            steps = steps[::-1]

        angle = math.atan2(steps[-1, 0] - steps[0, 0], steps[-1, 1] - steps[0, 1])
        return np.hstack((steps, np.array([angle] * len(steps)).reshape(-1, 1)))

    def process(self, dem, observer):
        yp, xp, angle = observer
        zp = dem[yp, xp] + self.observer_height
        viewshed = np.copy(dem)

        atan2deg = lambda x: (x if x >= 0 else (2 * math.pi) + x) * 360 / (2 * math.pi)

        def is_within(x, look, thres):
            flag = look < thres / 2 or look > 360 - thres / 2

            if not flag:
                return (look - (thres / 2)) <= x <= look + (thres / 2)
            else:
                l = look - thres / 2
                u = look + thres / 2

                if 360 > u >= 0 and x >= 0:
                    return (x <= u and x < 180) or (x > 180 and x > l % 360)
                else:
                    return (x >= l and x > 180) or (x < 180 and x < u % 360)

        # * Find perimiter
        h, w = dem.shape
        rr, cc = rectangle_perimeter((1, 1), end=(h - 2, w - 2), shape=dem.shape)

        # * Iterate through perimiter
        for yc, xc in zip(rr, cc):
            # * Form the line
            ray_y, ray_x = line(yp, xp, yc, xc)
            ray_z = dem[ray_y, ray_x]

            # * Check if line in sight
            line_angle = atan2deg(math.atan2(yc - yp, xc - xp))
            look_angle = atan2deg(angle)

            if not is_within(line_angle, look_angle, self.observer_view_angle):
                viewshed[ray_y, ray_x] = np.nan
                continue

            m = (ray_z - zp) / np.hypot(ray_y - yp, ray_x - xp)

            max_so_far = -np.inf
            for yi, xi, mi in zip(ray_y, ray_x, m):
                if mi < max_so_far:
                    viewshed[yi, xi] = np.nan
                else:
                    max_so_far = mi

        return viewshed

    def __call__(self, dem):
        IMAGE_SIZE = dem.shape[-1]
        # * Start by finding places of interest
        ref = Helper.get_slopes(
            dem.reshape(1, IMAGE_SIZE, IMAGE_SIZE), n=2, only_map=True
        )
        steps = self.get_poi(ref.reshape(IMAGE_SIZE, IMAGE_SIZE))

        # * Then move the observer but each viewshed will accumulate
        viewsheds = []
        for i, (yp, xp, ang) in enumerate(steps):
            viewshed = self.process(dem, (int(yp), int(xp), ang))
            if i > 0:
                viewshed[np.isnan(viewshed)] = viewsheds[-1][np.isnan(viewshed)]
            viewsheds.append(viewshed)

        return viewsheds


class Helper:
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_ranges(x):
        bmax = np.max(x.reshape(-1, x.shape[2] ** 2), axis=1)
        bmin = np.min(x.reshape(-1, x.shape[2] ** 2), axis=1)
        return bmax - bmin

    @staticmethod
    def get_percentage(x, of):
        side = x.shape[2] ** 2
        mins = np.min(x.reshape(-1, side), axis=1).reshape(-1, 1)
        return (
            np.count_nonzero(
                (x.reshape(-1, side) - np.repeat(mins, side, axis=1)) > of,
                axis=1,
            )
            / side
        )

    @staticmethod
    def get_skews(x):
        return stats.skew(x.reshape(-1, x.shape[2] ** 2), axis=1)

    @staticmethod
    def get_slopes(x, n=1, only_map=False):
        SH = x.shape[-1]

        _x = np.copy(x)
        _x[:, 1::2, :] = _x[:, 1::2, ::-1]
        xax = np.abs(np.diff(_x.reshape(-1, SH ** 2), n=n, axis=1))

        _y = np.copy(np.swapaxes(x, 1, 2))
        _y[:, 1::2, :] = _y[:, 1::2, ::-1]
        yax = np.abs(np.diff(_y.reshape(-1, SH ** 2), n=n, axis=1))

        nan_block = np.array([np.nan] * x.shape[0] * n).reshape(-1, n)

        xax = (np.concatenate((xax, nan_block), axis=1)).reshape(-1, SH, SH)
        xax[:, 1::2, :] = xax[:, 1::2, ::-1]

        yax = (np.concatenate((yax, nan_block), axis=1)).reshape(-1, SH, SH)
        yax[:, 1::2, :] = yax[:, 1::2, ::-1]
        yax = np.swapaxes(yax, 1, 2)

        zax = xax + yax
        if only_map:
            return zax
        return zax, np.nanmedian(zax.reshape(-1, SH ** 2), axis=1)


class TerrainDataset(Dataset):
    NAN = 0

    def __init__(
        self,
        dataset_glob,
        dataset_type,
        patch_size=30,
        sample_size=256,
        observer_pad=50,
        block_dimension=4,
        block_variance=4,
        observer_height=0.75,
        limit_samples=None,
        randomize=True,
        random_state=None,
        usable_portion=1.0,
        no_tqdm=False,
        transform=None,
    ):
        """
        dataset_glob -> glob to *.tif files (i.e. "data/MDRS/data/*.tif")
        dataset_type -> train or validation
        patch_size -> the 1m^2 area to read from .TIF
        sample_size -> the 0.1m^2 res area to be trained sample size
        observer_pad -> n pixels to pad before getting a random observer
        block_dimension -> Number of masks (motion step) for each submap
        block_variance -> how many different observer points
        observer_height -> Observer Height
        limit_samples -> Limit number of samples returned
        randomize -> predictable randomize
        random_state -> a value that gets added to seed
        usable_portion -> What % of the data will be used
        no_tqdm -> Disable tqdm progress
        transform -> if there is any, PyTorch Transforms
        """
        np.seterr(divide="ignore", invalid="ignore")

        # * Set Dataset attributes
        self.patch_size = patch_size
        self.sample_size = sample_size
        self.block_variance = block_variance
        self.observer_pad = observer_pad
        self.block_dimension = block_dimension

        # * PyTorch Related Variables
        self.transform = transform

        # * Gather files
        self.files = glob(dataset_glob)
        self.dataset_type = dataset_type
        self.usable_portion = usable_portion
        self.limit_samples = limit_samples

        self.randomize = randomize
        self.random_state = (
            random.randint(0, 100) if random_state is None else random_state
        )
        if self.randomize:
            random.seed(self.random_state)
            random.shuffle(self.files)

        # * Build dataset dictionary
        cache_name = os.path.dirname(__file__)
        cache_name += "/tmp/TDSDATA-"
        cache_name += f"RS{self.random_state}-IS{self.sample_size}-"
        cache_name += hashlib.md5(("".join(sorted(self.files))).encode()).hexdigest()

        if os.path.exists(cache_name):
            self.sample_dict = pickle.load(open(cache_name, "rb"))
        else:
            self.sample_dict = dict()
            start = 0
            for file in tqdm(self.files, ncols=100, disable=no_tqdm):
                blocks, mask = self.get_blocks(file, return_mask=True)

                if len(blocks[mask]) == 0:
                    print(f"Skipped file {file}")
                    continue

                self.sample_dict[file] = {
                    "start": start,
                    "end": start + len(blocks[mask]),
                    "mask": mask,
                    "min": np.min(blocks[mask]),
                    "max": np.max(blocks[mask]),
                    "range": np.max(Helper.get_ranges(blocks[mask])),
                }
                start += len(blocks[mask])
                del blocks

            pickle.dump(self.sample_dict, open(cache_name, "wb"))

        self.data_min = min(self.sample_dict.values(), key=lambda x: x["min"])["min"]
        self.data_max = max(self.sample_dict.values(), key=lambda x: x["max"])["max"]
        self.data_range = max(self.sample_dict.values(), key=lambda x: x["range"])[
            "range"
        ]

        # * Check if limit_samples is enough for this dataset
        if limit_samples is not None:
            assert (
                limit_samples <= self.get_len()
            ), "limit_samples cannot be bigger than dataset size"

        # * Viewshed Engine
        self.viewshed = Viewshed(
            observer_height=observer_height,
            observer_pad=self.observer_pad,
            observer_steps=self.block_dimension,
            data_range=self.data_range,
        )

        # * Dataset state
        self.current_file = None
        self.current_blocks = None

    def get_len(self):
        key = list(self.sample_dict.keys())[-1]
        return self.sample_dict[key]["end"]

    def __len__(self):
        if not self.limit_samples is None:
            return self.limit_samples
        return self.get_len()

    def __getitem__(self, idx):
        rel_idx = None
        for file, info in self.sample_dict.items():
            if idx >= info["start"] and idx < info["end"]:
                rel_idx = idx - info["start"]
                if self.current_file != file:
                    b = self.get_blocks(file)
                    self.current_blocks = b[info["mask"]]
                    self.current_file = file
                break

        current = np.copy(self.current_blocks[rel_idx])
        current -= np.min(current)
        current /= self.data_range

        adjusted = self.get_adjusted(current)
        viewshed = self.viewshed(adjusted)

        mask = np.isnan(viewshed).astype(np.uint8)
        mask = torch.from_numpy(mask).float()
        mask = mask.unsqueeze(1)

        adjusted = np.expand_dims(adjusted, axis=0)
        target = np.repeat(adjusted, self.block_dimension, axis=0)
        target = torch.from_numpy(target).float()
        target = target.unsqueeze(1)

        return target, mask

    def blockshaped(self, arr, nside):
        """
        Return an array of shape (n, nside, nside) where
        n * nside * nside = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        assert h % nside == 0, "{} rows is not evenly divisble by {}".format(h, nside)
        assert w % nside == 0, "{} cols is not evenly divisble by {}".format(w, nside)
        return (
            arr.reshape(h // nside, nside, -1, nside)
            .swapaxes(1, 2)
            .reshape(-1, nside, nside)
        )

    def get_adjusted(self, block):
        zoomed = zoom(block, 10, order=1)
        y, x = zoomed.shape
        startx = x // 2 - (self.sample_size // 2)
        starty = y // 2 - (self.sample_size // 2)
        return zoomed[
            starty : starty + self.sample_size, startx : startx + self.sample_size
        ]

    def get_blocks(self, file, return_mask=False):
        raster = rasterio.open(file)
        grid = raster.read(1)

        # Remove minimum
        grid[grid == np.min(grid)] = np.nan

        # Find the edges to cut from
        NL = np.count_nonzero(np.isnan(grid[:, 0]))
        NR = np.count_nonzero(np.isnan(grid[:, -1]))
        NT = np.count_nonzero(np.isnan(grid[0, :]))
        NB = np.count_nonzero(np.isnan(grid[-1, :]))

        w, h = grid.shape
        if NL > NR:
            grid = grid[w % self.patch_size : w, 0:h]
        else:
            grid = grid[0 : w - (w % self.patch_size), 0:h]

        w, h = grid.shape
        if NT > NB:
            grid = grid[0:w, h % self.patch_size : h]
        else:
            grid = grid[0:w, 0 : h - (h % self.patch_size)]

        blocks = self.blockshaped(grid, self.patch_size)

        # * Randomize
        if self.randomize:
            np.random.seed(self.random_state)
            np.random.shuffle(blocks)

        if self.dataset_type == "train":
            blocks = blocks[: int(len(blocks) * self.usable_portion)]
        else:
            blocks = blocks[int(len(blocks) * self.usable_portion) :]

        # * Add Variance
        blocks = np.repeat(blocks, self.block_variance, axis=0)

        if not return_mask:
            return blocks

        # * Remove blocks that contain nans
        base_mask = ~np.isnan(blocks).any(axis=1).any(axis=1)

        # * Further filter remeaning data in relation to z-score
        ranges = Helper.get_ranges(blocks[base_mask])
        mask = np.abs(stats.zscore(ranges)) < 2
        mask &= np.abs(stats.zscore(ranges)) > 0.05

        # * Eliminate unwanted ranges
        mask &= ranges >= 1
        mask &= ranges < 5

        # * Eliminate low terrains for observer
        mask &= Helper.get_percentage(blocks[base_mask], 1) > 0.2
        mask &= Helper.get_percentage(blocks[base_mask], 5) < 0.9

        # * Eliminate blocks by skewness in relation to z-score
        skews = Helper.get_skews(blocks[base_mask])
        mask &= np.abs(stats.zscore(skews)) < 1
        mask &= np.abs(stats.zscore(skews)) > 0.1

        # * Eliminate blocks by slopes in relation to z-score
        _, slopes = Helper.get_slopes(blocks[base_mask])
        mask &= np.abs(stats.zscore(slopes)) > 0.5

        base_mask[base_mask == True] &= mask

        return blocks, base_mask