"""Training a Diffusion Model with PyTorch."""

# pylint: disable=R0801

# Standard library
import socket

# Third-party
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch  # type: ignore
import xarray as xr
from denoising_diffusion_pytorch import GaussianDiffusion  # type: ignore
from denoising_diffusion_pytorch import Unet  # type: ignore


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation="nearest")


zoom = ""
hostname = socket.gethostname()
if "tsa" in hostname:
    data_train = xr.open_zarr(
        "/scratch/sadamov/pyprojects_data/aldernet/" + zoom + "/data_train.zarr"
    )
    data_valid = xr.open_zarr(
        "/scratch/sadamov/pyprojects_data/aldernet/" + zoom + "/data_valid.zarr"
    )
elif "nid" in hostname:
    data_train = xr.open_zarr(
        "/scratch/e1000/meteoswiss/scratch/sadamov/pyprojects_data/aldernet/"
        + zoom
        + "/data_train.zarr"
    )
    data_valid = xr.open_zarr(
        "/scratch/e1000/meteoswiss/scratch/sadamov/pyprojects_data/aldernet/"
        + zoom
        + "/data_valid.zarr"
    )

data_img = (
    data_train.ALNU.expand_dims({"channel": 1})
    .isel(y=slice(400, 464), x=slice(400, 464), valid_time=slice(0, 16))
    .transpose("valid_time", "channel", "y", "x")
    .to_numpy()
)

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = Unet(dim=64, dim_mults=(1, 2, 4, 8, 16, 32, 64), channels=1).cuda()
diffusion = GaussianDiffusion(
    model, image_size=64, timesteps=1000, loss_type="l1"
).cuda()

training_images = torch.from_numpy(data_img).cuda()  # pylint: disable=E1101
loss = diffusion(training_images).cuda()
loss.backward()

sampled_images = diffusion.sample(batch_size=4).cuda()

show(training_images.cpu()[0])
show(sampled_images.cpu()[0])

plt.imsave("sample1.png", sampled_images.cpu()[0][0, :, :])
plt.imsave("sample2.png", sampled_images.cpu()[1][0, :, :])
plt.imsave("sample3.png", sampled_images.cpu()[2][0, :, :])
plt.imsave("sample4.png", sampled_images.cpu()[3][0, :, :])

torch.cuda.empty_cache()
