# %% tags=["solution", "task"]
# ruff: noqa: F811
# %% [markdown] tags=[]
# # Exercise: Training COSDD
#
# In this section, we will train a COSDD model to remove row correlated and signal-dependent imaging noise. 
# You will load noisy data and examine the noise for spatial correlation, then initialise a model and monitor its training.
# Finally, you'll use the model to denoise the data.
#
# COSDD is a Ladder VAE with an autoregressive decoder -- a type of deep generative model. Deep generative models are trained with the objective of capturing all the structures and characteristics present in a dataset, i.e., modelling the dataset. In our case the dataset will be a collection of noisy microscopy images. 
#
# When COSDD is trained to model noisy images, it exploits differences between the structure of imaging noise and the structure of the clean signal to separate them, capturing each with different components of the model. Specifically, the noise will be captured by the autoregressive decoder and the signal will be captured by the VAE's latent variables. We can then feed an image into the model and sample a latent variable that will describe the image's clean signal content. This latent variable is then fed through a second network, which was trained alongside the main VAE, to reveal an estimate of the denoised image.

# %% [markdown] tags=[]
# <div class="alert alert-danger">
# Set your python kernel to <code>05_image_restoration</code>
# </div>

# %% tags=[]
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins.environments import LightningEnvironment
from tqdm import tqdm
import yaml

from COSDD import utils
from COSDD.models.get_models import get_models
from COSDD.models.hub import Hub
import interactive_plots as iplots

# %matplotlib inline

# %% tags=[]
assert torch.cuda.is_available()

# %% [markdown] tags=[]
# ## 1. Load the data
#
# In this example, we will be using the Mito Confocal dataset provided by: 
# Hagen, G.M., Bendesky, J., Machado, R., Nguyen, T.A., Kumar, T. and Ventura, J., 2021. Fluorescence microscopy datasets for training deep neural networks. GigaScience, 10(5), p.giab032.

# %% [markdown] tags=[]
# This data contains noise that is correlated along rows. We'll have a closer look at that later.
# For now, let's load it.

# %% [markdown] tags=[]
# <div class="alert alert-info">
#
# ### Task 1.1.
#
# The low signal-to-noise ratio data that we will be denoising has been downloaded and stored in the `/mnt/efs/aimbl_2025/data/` directory as `mito-confocal-lowsnr.tif`. We will load it in the following cell using `utils.load_data`. This requires four arguments that are described below. `paths`, `axes` and `n_dimensions` have already been entered. 
#
# Enter the file name for `patterns`.
# </div>

# %% [markdown] tags=[]
# `paths` (str): Path to the directory the training data is stored in. Can be a list of strings if using more than one directory.
#
# `patterns` (str): glob pattern to identify files within `paths` that will be used as training data.
#
# `axes` (str): (S(ample) | C(hannel) | T(ime) | Z | Y | X). The meaning of each axis in the loaded data, e.g., for a stack of images "SYX". 
#
# `n_dimensions` (int): Number of spatial dimensions of your data, i.e, 1 for time series, 2 for images, 3 for volumes.

# %% tags=["task"]
# load the data
paths = "/mnt/efs/aimbl_2025/data/"
patterns = ... # Enter the data's file name here
axes = "SYX"
n_dimensions = 2
low_snr, original_sizes = utils.load_data(
    paths=paths, patterns=patterns, axes=axes, n_dimensions=n_dimensions
)

# %% tags=["solution"]
# load the data
paths = "/mnt/efs/aimbl_2025/data/"
patterns = "mito-confocal-lowsnr.tif"
axes = "SYX"
n_dimensions = 2
low_snr, original_sizes = utils.load_data(
    paths=paths, patterns=patterns, axes=axes, n_dimensions=n_dimensions
)

# %% [markdown] tags=[]
# <div class="alert alert-info">
#
# ### Task 1.2.
#
# The data is held in the `low_snr` variable. 
#
# Check the shape and data type using `.shape` and `.dtype`.
#
# The shape should be of the format: (Number of images, Number of channels, Height, Width), and the data type should be float32.
#
# </div>

# %% tags=["task"]
print(f"Noisy data size: {low_snr...}")  # Replace ... with the correct attribute
print(f"Noisy data dtype: {low_snr...}")  # Replace ... with the correct attribute

# %% tags=["solution"]
print(f"Noisy data size: {low_snr.shape}")
print(f"Noisy data dtype: {low_snr.dtype}")

# %% [markdown] tags=[]
# ## 2. Examine spatial correlation of the noise

# %% [markdown] tags=[]
# COSDD can be applied to noise that is correlated along rows or columns of pixels (or not spatially correlated at all).
# However, it cannot be applied to noise that is correlated along rows *and* columns of pixels.
# Noise2Void is designed for noise that is not spatially correlated at all.
#
# When we say that the noise is spatially correlated, we mean that knowing the value of the noise in one pixel tells us something about the noise in other (usually nearby) pixels.
# Specifically, positive correlatation between two pixels tells us that if the intensity of the noise value in one pixel is high, the intensity of the noise value in the other pixel is likely to be high.
# Similarly, if one is low, the other is likely to be low.
# Negative correlation between pixels means that a low noise intensity in one pixel is more likely if the intensity in the other is high, and vice versa.
#
# To examine an image's spatial correlation, we can create an autocorrelation plot. 
# The plot will have two axes, horizontal lag and vertical lag, and it tells us what the correlation between a pair of pixels separated by a given horizontal and vertical lag is.
# For example, if the square at a horizontal lag of 3 and a vertical lag of 6 is red, it means that if we picked any pixel in the image, then counted 3 pixels to the right and 6 pixels down, this pair of pixels is positively correlated.
# Correlation is symmetric, so the same is true if we counted left or up.

# %% [markdown] tags=[]
# <div class="alert alert-warning">
#
# ### Question 2.1.
#
# Below are three examples of noise. Beneath each is an autocorrelation plot showing how they are spatially correlated.
# Identify which noise examples could be removed by:<br>
# (a) COSDD<br>
# (b) Noise2Void<br>
# (c) neither
# </div>

# %% [markdown] tags=[]
# <img src="resources/ac-question.png"/>

# %% [markdown] tags=["solution"]
# 1: COSDD and Noise2Void<br>
# 2: COSDD<br>
# 3: Neither

# %% [markdown] tags=[]
# <div class="alert alert-info">
#
# ### Task 2.1.
#
# Now we will create an autocorrelation plot of the noise in the data we loaded.
# To do this, we need a sample of pure noise.
# This can be a dark patch of the background in `low_snr`. 
#
# Run the cell below to start looking at the data.
# Adjust the sliders for `Image index`, `Top`, `Bottom`, `Left` and `Right` to explore crops of the data and identify a suitable background patch.
# When decided, click `calculate autocorrelation`.
# Your autocorrelation plot should report only horizontal correlation.
# </div>

# %% tags=[]
layout = iplots.find_dark_patch(low_snr)
display(layout)

# %% [markdown] tags=[]
# In the autocorrelation plot, all of the squares should be white except for the top row. The autocorrelation of the square at (0, 0) will always be 1.0 because a pixel's value will always be perfectly correlated with itself. We define this type of noise as correlated along the x axis.
#
# This is the type of noise that COSDD is designed to remove.
# Note that COSDD would still work if the data contained spatially *un*correlated noise.

# %% [markdown] tags=[]
# ## 3. Create dataloaders

# %% [markdown] tags=[]
# Now that we're familar with our data, we can get it into a dataloader ready to train a denoiser.

# %% [markdown] tags=[]
# <div class="alert alert-info">
#
# ### Task 3.1.
#
# We will use `utils.DataModule` to prepare the dataloaders. This has four arguments that are described below.
# Three have already been set.
#
# Set `train_split` such that 90% of the images will be used as a training set and 10% used as a validation set.
#
# </div>

# %% [markdown] tags=[]
# `batch_size` (int): Number of images passed through the network at a time. <br>
# `n_grad_batches` (int): Number of batches to pass through the network before updating parameters. <br>
# `crop_size` (tuple(int)): The size of randomly cropped patches. Should be less than the dimensions of your images. <br>
# `train_split` (0 < float < 1): Fraction of images to be used in the training set, with the remainder used for the validation set.
#

# %% tags=["task"]
real_batch_size = 4
n_grad_batches = 4
print(f"Effective batch size: {real_batch_size * n_grad_batches}")
crop_size = (256, 256)
train_split = ...  # Enter a training split here

datamodule = utils.DataModule(
    low_snr=low_snr,
    batch_size=real_batch_size,
    rand_crop_size=crop_size,
    train_split=train_split,
)

# %% tags=["solution"]
real_batch_size = 4
n_grad_batches = 4
print(f"Effective batch size: {real_batch_size * n_grad_batches}")
crop_size = (256, 256)
train_split = 0.9

datamodule = utils.DataModule(
    low_snr=low_snr,
    batch_size=real_batch_size,
    rand_crop_size=crop_size,
    train_split=train_split,
)

# %% [markdown] tags=[]
# <div class="alert alert-success">
#
# ### Checkpoint 1
# With our data ready, we can use it to train a COSDD model for denoising.
#
# </div>

# %% [markdown] tags=[]
# ## 4. Create the model

# %% [markdown] tags=[]
# <img src="resources/explainer.png"/>
#
# COSDD is a Variational Autoencoder (solid arrows) trained to model the distribution of noisy images $\mathbf{x}$. 
#
# a) 
# The autoregressive (AR) decoder models the noise component of the images, while the latent variable models only the clean signal component $\mathbf{s}$.
# In a second step (dashed arrows), the *signal decoder* is trained to map latent variables into image space, producing an estimate of the signal underlying $\mathbf{x}$.
#
# b)
# To ensure that the decoder models only the imaging noise and the latent variables capture only the signal, the AR decoder's receptive field is modified.
# In a full AR receptive field, each output pixel (red) is a function of all input pixels located above and to the left (blue). In our decoder's row-based AR receptive field, each output pixel is a function of input pixels located in the same row, which corresponds to the row-correlated structure of imaging noise.

# %% [markdown] tags=[]
# <div class="alert alert-info">
#
# ### Task 4.1.
#
# The model we will train to denoise consists of four modules.
# Each has it's own hyper-parameters.
# Most have been set to defaults for a small model.
#
# There are two hyperparameters that need to be set:
# 1) `noise_direction`. This tells the model which axis our noise is correlated along. It needs to be set to either `x`, `y` or `z`. Look at the autocorrelation plot to set the correct value.
# 2) `use_direct_denoiser`. Setting this to `True` will slightly slow down training but, once the model is trained, will massively speed up denoising. Set to your preference of `True` or `False`.
#
# </div>

# %% [markdown] tags=[]
# `lvae` The ladder variational autoencoder that will output latent variables.
# * `s_code_channels` (int): Number of channels in outputted latent variable.
# * `n_layers` (int): Number of levels in the ladder vae.
#
# `ar_decoder` The autoregressive decoder that will decode latent variables into a distribution over the input.
# * `noise_direction` (str): Axis along which noise is correlated: `"x"`, `"y"` or `"z"`. This needs to match the orientation of the noise structures we revealed in the autocorrelation plot in Task 1.2.
# * `n_gaussians` (int): Number of components in Gaussian mixture used to model data.
#
# `direct_denoiser` The U-Net that can optionally be trained to predict the MMSE or MMAE of the denoised images. This will slow training slightly but massively speed up inference and is worthwile if you have an inference dataset in the gigabytes. See [this paper](https://arxiv.org/abs/2310.18116). Enable or disable the direct denoiser by setting `use_direct_denoiser` to `True` or `False`.
# * `loss_fn` (str): Whether to use `"L1"` or `"MSE"` loss function to predict either the mean or pixel-wise median of denoised images respectively.
#
# `hub` The hub that will unify and train the above modules.
# * `gradient_checkpoints` (bool): Whether to use gradient checkpointing during training. This reduces memory consumption but increases training time.

# %% tags=["task"]
s_code_channels = 64
n_layers = 6
noise_direction = ...  # 
n_gaussians = 10
use_direct_denoiser = ...  # 
dd_loss_fn = "MSE"
graident_checkpoints = False

config = {
    "data": {
        "number-dimensions": n_dimensions,
    },
    "train-parameters": {
        "number-grad-batches": n_grad_batches,
        "use-direct-denoiser": use_direct_denoiser,
        "direct-denoiser-loss": dd_loss_fn,
        "crop-size": crop_size,
    },
    "hyper-parameters": {
        "s-code-channels": s_code_channels,
        "number-layers": n_layers,
        "number-gaussians": n_gaussians,
        "noise-direction": noise_direction,
    },
}
config = utils.get_defaults(config)

lvae, ar_decoder, s_decoder, direct_denoiser = get_models(config, n_channels=low_snr.shape[1])

data_mean = low_snr.mean()
data_std = low_snr.std()
hub = Hub(
    vae=lvae,
    ar_decoder=ar_decoder,
    s_decoder=s_decoder,
    direct_denoiser=direct_denoiser,
    data_mean=data_mean,
    data_std=data_std,
    n_grad_batches=n_grad_batches,
    checkpointed=graident_checkpoints,
)

# %% tags=["solution"]
s_code_channels = 64
n_layers = 6
noise_direction = "x"
n_gaussians = 10
use_direct_denoiser = True
dd_loss_fn = "MSE"
graident_checkpoints = False

config = {
    "data": {
        "number-dimensions": n_dimensions,
    },
    "train-parameters": {
        "number-grad-batches": n_grad_batches,
        "use-direct-denoiser": use_direct_denoiser,
        "direct-denoiser-loss": dd_loss_fn,
        "crop-size": crop_size,
    },
    "hyper-parameters": {
        "s-code-channels": s_code_channels,
        "number-layers": n_layers,
        "number-gaussians": n_gaussians,
        "noise-direction": noise_direction,
    },
}
config = utils.get_defaults(config)

lvae, ar_decoder, s_decoder, direct_denoiser = get_models(config, n_channels=low_snr.shape[1])

data_mean = low_snr.mean()
data_std = low_snr.std()
hub = Hub(
    vae=lvae,
    ar_decoder=ar_decoder,
    s_decoder=s_decoder,
    direct_denoiser=direct_denoiser,
    data_mean=data_mean,
    data_std=data_std,
    n_grad_batches=n_grad_batches,
    checkpointed=graident_checkpoints,
)

# %% [markdown] tags=[]
# ## 5. Training the model

# %% [markdown] tags=[]
# <div class="alert alert-info">
#
# ### Task 5.1.
#
# Open Tensorboard to monitor training. (See Task 3 of 01_CARE).
# Choose `03_COSDD/checkpoints` for the folder.
# In there you'll see the training logs of a model that was trained for about 1.75 hours.
#
# Unlike CARE, this model has more than one loss curve.
# The cell below describes how to interpret each one.
#
# We're going to train our model for only 15 minutes, but we should see the training logs start to follow this previous model.
# </div>

# %% [markdown] tags=[]
# #### Tensorboard metrics
#
# In the SCALARS tab, there will be 4 metrics to track (5 if direct denoiser is enabled). These are:
# 1. `kl_div` The Kullback-Leibler divergence between the VAE's approximate posterior and its prior. This can be thought of as a measure of how much information about the input image is going into the VAE's latent variables. We want information about the input's underlying clean signal to go into the latent variables, so this metric shouldn't go all the way to zero. Instead, it can typically go either up or down during training before plateauing.
# 2. `nll` The negative log-likelihood of the AR decoder's predicted distribution given the input data. This is how accurately the AR decoder is able to predict the input. This value can go below zero and should decrease throughout training before plateauing.
# 3. `elbo` The Evidence Lower Bound, which is the total loss of the main VAE. This is the sum of the kl and reconstruction loss and should decrease throughout training before plateauing.
# 4. `sd_loss` The mean squared error between the noisy image and the image predicted by the signal decoder. This metric should steadily decrease towards zero without ever reaching it. Sometimes the loss will not go down for the first few epochs because its input (produced by the VAE) is rapidly changing. This is ok and the loss should start to decrease when the VAE stabilises. 
# 5. `dd_loss` The mean squared error between the output of the direct denoiser and the clean images predicted by the signal decoder. This will only be present if `use_direct_denoiser` is set to `True`. The metric should steadily decrease towards zero without ever reaching it, but may be unstable at the start of training as its targets (produced by the signal decoder) are rapidly changing.
#
# There will also be an IMAGES tab. This shows noisy input images from the validation set and some outputs. These will be two randomly sampled denoised images (sample 1 and sample 2), the average of ten denoised images (mmse) and if the direct denoiser is enabled, its output (direct estimate).
#
# If noise has not been fully removed from the output images, try increasing `n_gaussians` argument of the AR decoder. This will give it more flexibility to model complex noise characteristics. However, setting the value too high can lead to unstable training. Typically, values from 3 to 5 work best.
#
# Note that the trainer is set to train for only 15 minutes in this example. Remove the line with `max_time` to train fully.

# %% [markdown] tags=[]
# <div class="alert alert-info">
#
# ### Task 5.2.
#
# In the following cell, set a sensible `model_name`. You will use this to recall the trained model later.
#
# Run the cell after to start training.
#
# The `max_time` parameter in the cell below means we'll only train the model for 15 minutes, just to get idea of what to expect. In the future, to remove the time restriction, the `max_time` parameter can be set to `None`.
# </div>

# %% [markdown] tags=[]
# `model_name` (str): Should be set to something appropriate so that the trained parameters can be used later for inference.
#
# `max_epochs` (int): The number of training epochs.
#
# `patience` (int): If the validation loss has plateaued for this many epochs, training will stop.
#
# `max_time` (str): Maximum time to train for. Must be of form "DD:HH:MM:SS", or just `None`.

# %% tags=["task"]
model_name = ...  # Enter a model name here
max_epochs = 250
patience = 50
max_time = "00:00:15:00"
gpu_idx = [0]

checkpoint_path = os.path.join("checkpoints", model_name)
logger = TensorBoardLogger(checkpoint_path)

trainer = pl.Trainer(
    logger=logger,
    accelerator="gpu",
    devices=gpu_idx,
    max_epochs=max_epochs,
    max_time=max_time,
    callbacks=[EarlyStopping(patience=patience, monitor="elbo/val")],
    precision="16",
    plugins=[LightningEnvironment()],
)

# %% tags=["solution"]
model_name = "mito-confocal"
max_epochs = 250
patience = 50
max_time = "00:00:15:00"
gpu_idx = [0]

checkpoint_path = os.path.join("checkpoints", model_name)
logger = TensorBoardLogger(checkpoint_path)

trainer = pl.Trainer(
    logger=logger,
    accelerator="gpu",
    devices=gpu_idx,
    max_epochs=max_epochs,
    max_time=max_time,
    callbacks=[EarlyStopping(patience=patience, monitor="elbo/val")],
    precision="16",
    plugins=[LightningEnvironment()],
)

# %% tags=[]
try:
    trainer.fit(hub, datamodule=datamodule)
except KeyboardInterrupt:
    print("KeyboardInterupt")
finally:
    # Save trained model
    trainer.save_checkpoint(os.path.join(checkpoint_path, f"final_model.ckpt"))
    with open(os.path.join(checkpoint_path, 'training-config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
torch.cuda.empty_cache()

# %% [markdown] tags=[]
# <div class="alert alert-success">
#
# ## Checkpoint 2
# We've now trained a COSDD model to denoise our data. Continue to the next part to use it to get some results.
#
# </div>

# %% [markdown] tags=[]
# ## 6. Load test data
# The images that we want to denoise are loaded here. These are the same that we used for training, but we'll only load 3 to speed up inference.
#
# We'll also get them into a dataloader.

# %% tags=[]
# load the data
paths = "/mnt/efs/aimbl_2025/data/"
patterns = "mito-confocal-lowsnr.tif"
axes = "SYX"
n_dimensions = 2
test_data, original_sizes = utils.load_data(
    paths=paths, patterns=patterns, axes=axes, n_dimensions=n_dimensions
)
test_data = test_data[:3]
print(f"Test data size: {test_data.size()}")

predict_batch_size = 1

predict_set = utils.PredictDataset(low_snr)
predict_loader = torch.utils.data.DataLoader(
    predict_set,
    batch_size=predict_batch_size,
    shuffle=False,
    pin_memory=True,
)

# %% [markdown] tags=[]
# ## 7. Load trained model

# %% [markdown] tags=[]
# <div class="alert alert-info">
#
# ### Task 7.1.
#
# Our model was only trained for 15 minutes. This is long enough to get some denoising results, but a model trained for longer would do a lot better. In the cell below, load the trained model by recalling the value you gave for `model_name`. Then procede through the notebook to look at how well it performs. 
#
# Once you reach the end of the notebook, return to this cell to load a model that has been trained for 1.75 hours by uncommenting line 3, then run the notebook again to see how much difference the extra training time makes. 
# </div>

# %% tags=["task"]
model_name = ...  ### Insert the model name here
checkpoint_path = os.path.join("checkpoints", model_name)
# checkpoint_path = "checkpoints/mito-confocal-pretrained" ### Once you reach the bottom of the notebook, return here and uncomment this line to see the pretrained model

with open(os.path.join(checkpoint_path, "training-config.yaml")) as f:
    train_cfg = yaml.load(f, Loader=yaml.FullLoader)

lvae, ar_decoder, s_decoder, direct_denoiser = get_models(train_cfg, low_snr.shape[1])
hub = Hub.load_from_checkpoint(
    os.path.join(checkpoint_path, "final_model.ckpt"),
    vae=lvae,
    ar_decoder=ar_decoder,
    s_decoder=s_decoder,
    direct_denoiser=direct_denoiser,
)

gpu_idx = [0]
predictor = pl.Trainer(
    accelerator="gpu",
    devices=gpu_idx,
    enable_progress_bar=False,
    enable_checkpointing=False,
    logger=False,
    precision="32",
    plugins=[LightningEnvironment()],
)

# %% tags=["solution"]
model_name = "mito-confocal"
checkpoint_path = os.path.join("checkpoints", model_name)
# checkpoint_path = "checkpoints/mito-confocal-pretrained" ### Once you reach the bottom of the notebook, return here and uncomment this line to see the pretrained model

with open(os.path.join(checkpoint_path, "training-config.yaml")) as f:
    train_cfg = yaml.load(f, Loader=yaml.FullLoader)

lvae, ar_decoder, s_decoder, direct_denoiser = get_models(train_cfg, low_snr.shape[1])
hub = Hub.load_from_checkpoint(
    os.path.join(checkpoint_path, "final_model.ckpt"),
    vae=lvae,
    ar_decoder=ar_decoder,
    s_decoder=s_decoder,
    direct_denoiser=direct_denoiser,
)

gpu_idx = [0]
predictor = pl.Trainer(
    accelerator="gpu",
    devices=gpu_idx,
    enable_progress_bar=False,
    enable_checkpointing=False,
    logger=False,
    precision="bf16-mixed",
    plugins=[LightningEnvironment()],
)

# %% [markdown] tags=[]
# ## 8. Denoise
# In this section, we will look at how COSDD does inference. 
#
# The model denoises images randomly, giving us a different output each time. First, we will compare three randomly sampled denoised images for the same noisy image. Then, we will produce a single consensus estimate by averaging randomly sampled denoised images. Finally, if the Direct Denoiser was trained in the previous step, we will see how it can be used to estimate this average in a single pass.

# %% [markdown] tags=[]
# ### 8.1 Random sampling 
# First, we will denoise each image three times and look at the difference between each estimate.

# %% tags=[]
use_direct_denoiser = False
n_samples = 7

hub.direct_pred = use_direct_denoiser
samples = []
for _ in tqdm(range(n_samples)):
    out = predictor.predict(hub, predict_loader)
    out = torch.cat(out, dim=0)
    samples.append(out)

samples = torch.stack(samples, dim=1).half()

# %% [markdown] tags=[]
# <div class="alert alert-info">
#
# ### Task 8.1.
#
# Here, we'll look at the original noisy image and the random denoised estimates. Use the sliders to look at different images and adjust the crop. 
# If you're using the model trained in 15 minutes, don't worry if the results don't look very good. 
#
# Use this section to really explore the results. Compare high intensity reigons to low intensity reigons, zoom in and out and spot the differences between the different samples. 
#
# The sampled denoised images have differences that express the uncertainty involved in this denoising problem.
# </div>

# %% tags=[]
layout = iplots.plot_samples(test_data, samples)
display(layout)

# %% [markdown] tags=[]
# ### 8.2 MMSE estimate
#
# In the next cell, we sample many denoised images and average them for the minimum mean square estimate (MMSE). The averaged images will be stored in the `MMSEs` variable, which has the same dimensions as `low_snr`.

# %% [markdown] tags=[]
# <div class="alert alert-info">
#
# ### Task 8.2.
#
# In the next cell, we will sample 10 randomly denoised estimates. 
# Explore their average - the MMSE estimate - to understand the smoothing effect of averaging so many samples.
# </div>

# %% tags=[]
use_direct_denoiser = False
n_samples = 10

hub.direct_pred = use_direct_denoiser

samples = []
for _ in tqdm(range(n_samples)):
    out = predictor.predict(hub, predict_loader)
    out = torch.cat(out, dim=0)
    samples.append(out)

samples = torch.stack(samples, dim=1).half()
MMSEs = torch.mean(samples, dim=1)

# %% tags=[]
layout = iplots.plot_mmse(test_data, MMSEs, samples)
display(layout)


# %% [markdown] tags=[]
# The MMSE will usually be closer to the reference than an individual sample and would score a higher PSNR, although it will also be blurrier.

# %% [markdown] tags=[]
# ### 8.3 Direct denoising
# Sampling 10 or more images and averaging them is a very time consuming. If the direct denoiser was trained in a previous step, it can be used to directly output what the average denoised image would be for a given noisy image.

# %% [markdown] tags=[]
# <div class="alert alert-info">
#
# ### Task 8.3.
#
# Did you enable the direct denoiser before training? If so, set `use_direct_denoiser` to `True` to use the Direct Denoiser for inference. If not, go back to Section 7 to load the pretrained model and return here. 
#
# Notice how much quicker the direct denoiser is than generating the MMSE results. Visually inspect and explore the results in the same way as before.
# </div>

# %% tags=["task"]
use_direct_denoiser = ...  # Enter True or False here
hub.direct_pred = use_direct_denoiser

direct = predictor.predict(hub, predict_loader)
direct = torch.cat(direct, dim=0).half()

# %% tags=["solution"]
use_direct_denoiser = True
hub.direct_pred = use_direct_denoiser

direct = predictor.predict(hub, predict_loader)
direct = torch.cat(direct, dim=0).half()

# %% tags=[]
layout = iplots.plot_direct(test_data, direct, MMSEs)
display(layout)

# %% [markdown] tags=[]
# <div class="alert alert-info">
#
# ### Task 8.4.
#
# If you haven't already, return to Task 7.1, uncomment line three and look at the results from a denoiser that was trained for 1.75 hours.
# </div>

# %% [markdown] tags=[]
# ### 9. Incorrect receptive field
#
# Earlier, when preparing the model, we told it that the noise was correlated along the x axis. 
# If we had instead told it the noise was correlated along the y axis, denoising would have failed.
#
# These images show what that would look like.

# %% [markdown] tags=[]
# <img src="./resources/penicillium_ynm.png">

# %% [markdown] tags=[]
# <div class="alert alert-success">
#
# ## Checkpoint 3
#
# We've completed the process of training and applying a COSDD model for denoising, but there's still more it can do. Optionally continue to the bonus notebook, bonus-exercise.ipynb, to see how the model of the data can be used to generate new clean and noisy images.
#
# </div>

# %% tags=[]