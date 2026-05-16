# %% [markdown] tags=["solution", "task"]
# ruff: noqa: F811
# %% [markdown]
# # Bonus exercise. Generating new images with COSDD
#
# COSDD is a deep generative model that captures the structures and characteristics of our data. In this notebook, we'll see how accurately it can represent our training data, in both the signal and the noise. We'll do this by using the model to generate entirely new images. These will be images that look like the ones in our training data but don't actually exist. This is the same as how models like DALL-E can generate entirely new images.

# %% [markdown]
# <div class="alert alert-danger">
# Set your python kernel to <code>05_image_restoration</code>
# </div>

# %%
import os
import yaml

import torch
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interactive_output
import ipywidgets as widgets

from COSDD import utils
from COSDD.models.get_models import get_models
from COSDD.models.hub import Hub

# %matplotlib inline

# %%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# %% [markdown]
# ## 1. Load data

# %%
# load the data
paths = "/mnt/efs/aimbl_2025/data/"
patterns = "mito-confocal-lowsnr.tif"
axes = "SYX"
n_dimensions = 2
low_snr, original_sizes = utils.load_data(
    paths=paths, patterns=patterns, axes=axes, n_dimensions=n_dimensions
)

# %% [markdown]
# <div class="alert alert-info">
#
# ### Task 1.1.
#
# Load the model trained in the first notebook by entering your `model_name`, or alternatively, uncomment line 4 to load the pretrained model.
# </div>

# %% tags=["task"]
model_name = ...  # Enter the model name here
checkpoint_path = os.path.join("checkpoints", model_name)

# checkpoint_path = "checkpoints/mito-pretrained"

with open(os.path.join(checkpoint_path, "training-config.yaml")) as f:
    train_cfg = yaml.load(f, Loader=yaml.FullLoader)

lvae, ar_decoder, s_decoder, direct_denoiser = get_models(train_cfg, low_snr.shape[1])

hub = Hub.load_from_checkpoint(
    os.path.join(checkpoint_path, "final_model.ckpt"),
    vae=lvae,
    ar_decoder=ar_decoder,
    s_decoder=s_decoder,
    direct_denoiser=direct_denoiser,
).to(device)

# %% tags=["solution"]
model_name = "mito-confocal"
checkpoint_path = os.path.join("checkpoints", model_name)

# checkpoint_path = "checkpoints/mito-confocal-pretrained"

with open(os.path.join(checkpoint_path, "training-config.yaml")) as f:
    train_cfg = yaml.load(f, Loader=yaml.FullLoader)

lvae, ar_decoder, s_decoder, direct_denoiser = get_models(train_cfg, low_snr.shape[1])

hub = Hub.load_from_checkpoint(
    os.path.join(checkpoint_path, "final_model.ckpt"),
    vae=lvae,
    ar_decoder=ar_decoder,
    s_decoder=s_decoder,
    direct_denoiser=direct_denoiser,
).to(device)

# %% [markdown]
# ## 2. Generating new noise for a real noisy image
#
# First, we'll pass a noisy image to the VAE and generate a random sample from the AR decoder. This will give us another noisy image with the same underlying clean signal but a different random sample of noise.

# %%
inp_image = low_snr[:1, :, :512, :512].cuda()
reconstructions = hub.reconstruct(inp_image)
denoised = reconstructions["s_hat"].cpu()
noisy = reconstructions["x_hat"].cpu()

# %% [markdown]
# <div class="alert alert-info">
#
# ### Task 2.1.
#
# Now we will look at the original noisy image and the generated noisy image. Adjust `top`, `bottom`, `left` and `right` to view different crops of the reconstructed image.
#
# </div>

# %%
vmin = np.percentile(inp_image.cpu(), 1)
vmax = np.percentile(inp_image.cpu(), 99)

num_channels = inp_image.shape[1]
max_height = inp_image.shape[2]
max_width = inp_image.shape[3]

vertical_widget = widgets.IntRangeSlider(
    description="Vertical crop",
    min=0,
    max=max_height,
    step=1,
    value=[0, max_height],
    orientation="vertical",
    style={"description_width": "initial"},
    layout=widgets.Layout(margin='0 0 0 30px')
)
horizontal_widget = widgets.IntRangeSlider(
    description="Horizontal crop",
    min=0,
    max=max_width,
    step=1,
    value=[0, max_width],
    style={"description_width": "initial"},
)

### Explore slices of the data here
def plot_crop(horizontal, vertical):
    left, right = horizontal[0], horizontal[1]
    top, bottom = vertical[1], vertical[0]
    top = max_height - top
    bottom = max_width - bottom
    crop = (0, slice(top, bottom), slice(left, right))
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(inp_image[0][crop].cpu(), vmin=vmin, vmax=vmax)
    ax[0].set_title("Original noisy image")
    ax[1].imshow(noisy[0][crop], vmin=vmin, vmax=vmax)
    ax[1].set_title("Generated noisy image")
    ax[2].imshow(denoised[0][crop], vmin=vmin, vmax=vmax)
    ax[2].set_title("Denoised image")
    plt.show()

interactive_output_widget = interactive_output(
    plot_crop,
    {
        "horizontal": horizontal_widget,
        "vertical": vertical_widget,
    },
)

sliders = widgets.HBox([horizontal_widget, vertical_widget])

layout = widgets.VBox([sliders, interactive_output_widget])

display(layout)

# %% [markdown]
# The spatial correlation of the generated noise can be compared to that of the real noise to get an idea of how accurate the model is. Since we have the denoised version of the generated image, we can get a noise sample by just subtracting it from the noisy versions.

# %%
real_noise = low_snr[8, 0, 800:, 800:]
generated_noise = noisy[0, 0] - denoised[0, 0]

real_ac = utils.autocorrelation(real_noise, max_lag=25)
generated_ac = utils.autocorrelation(generated_noise, max_lag=25)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ac1 = ax[0].imshow(real_ac, cmap="seismic", vmin=-1, vmax=1)
ax[0].set_title("Autocorrelation of real noise")
ax[0].set_xlabel("Horizontal lag")
ax[0].set_ylabel("Vertical lag")
ac2 = ax[1].imshow(generated_ac, cmap="seismic", vmin=-1, vmax=1)
ax[1].set_title("Autocorrelation of generated noise")
ax[1].set_xlabel("Horizontal lag")
ax[1].set_ylabel("Vertical lag")

fig.colorbar(ac2, fraction=0.045)
plt.show()

# %% [markdown]
# ## 3. Generating new images
#
# This time, we'll take a sample from the VAE's prior. This will be a latent variable containing information about a brand new signal. The signal decoder will take that latent variable and convert it into a clean image. The AR decoder will take the latent variable and create an image with the same clean image plus noise.

# %% [markdown]
# <div class="alert alert-info">
#
# ### Task 3.1.
#
# Set the `n_imgs` variable below to decide how many images to generate. If you set it too high you'll get an out-of-memory error, but don't worry, just restart the kernel and run again with a lower value.
#
# Explore the images you generated in the second cell below. Look at the differences between them to see what aspects of the signal the model has learned to generate.
#
# </div>

# %% tags=["task"]
n_imgs = ... # Insert an integer here
generations = hub.sample_prior(n_imgs=n_imgs)
new_denoised = generations["s"].cpu()
new_noisy = generations["x"].cpu()

# %% tags=["solution"]
n_imgs = 5 # Insert an integer here
generations = hub.sample_prior(n_imgs=n_imgs)
new_denoised = generations["s"].cpu()
new_noisy = generations["x"].cpu()

# %%
vmin = np.percentile(new_noisy.cpu(), 1)
vmax = np.percentile(new_noisy.cpu(), 99)

num_images = new_noisy.shape[0]
num_channels = new_noisy.shape[1]
max_height = new_noisy.shape[2]
max_width = new_noisy.shape[3]

index_slider = widgets.BoundedIntText(
    description="Image index: ", min=0, max=num_images-1, step=1, value=0
)

vertical_widget = widgets.IntRangeSlider(
    description="Vertical crop",
    min=0,
    max=max_height,
    step=1,
    value=[0, max_height],
    orientation="vertical",
    style={"description_width": "initial"},
    layout=widgets.Layout(margin='0 0 0 30px')
)
horizontal_widget = widgets.IntRangeSlider(
    description="Horizontal crop",
    min=0,
    max=max_width,
    step=1,
    value=[0, max_width],
    style={"description_width": "initial"},
)

### Explore slices of the data here
def plot_crop(image_index, horizontal, vertical):
    left, right = horizontal[0], horizontal[1]
    top, bottom = vertical[1], vertical[0]
    top = max_height - top
    bottom = max_width - bottom
    crop = (0, slice(top, bottom), slice(left, right))
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(new_noisy[image_index][crop], vmin=vmin, vmax=vmax)
    ax[0].set_title("Generated noisy image")
    ax[1].imshow(new_denoised[image_index][crop], vmin=vmin, vmax=vmax)
    ax[1].set_title("Generated clean image")
    plt.show()

interactive_output_widget = interactive_output(
    plot_crop,
    {
        "image_index": index_slider,
        "horizontal": horizontal_widget,
        "vertical": vertical_widget,
    },
)

sliders = widgets.HBox([horizontal_widget, vertical_widget])
slide_and_index = widgets.VBox([index_slider, sliders])

layout = widgets.VBox([slide_and_index, interactive_output_widget])

display(layout)

# %% [markdown]
# <div class="alert alert-success">
#
# ### Checkpoint 3
#
# In this notebook, we saw how the model you trained in the first notebook has learned to describe the data. We first added a new sample of noise to an existing noisy image. We then generated a clean image that looks like it could be from the training data but doesn't actually exist. <br>
# You can now optionally return to section 3.1 to load a model that's been trained for much longer, otherwise, you've finished this module on COSDD.
#
# </div>

# %%
