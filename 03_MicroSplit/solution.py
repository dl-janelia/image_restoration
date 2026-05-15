# %% tags=["solution", "task"]
# ruff: noqa: F811
# %% [markdown] tags=[]
# # MicroSplit: Semantic Unmixing of Fluorescent Microscopy Data

# TODO: reduce dset size!!!
#
# In this notebook, you will work with MicroSplit, a deep learning-based computational multiplexing technique that allows imaging multiple cellular structures within a single fluorescent channel. The method enables imaging more cellular structures, imaging them faster, and at reduced overall light exposure.
#
# <p>
#     <img src="imgs/Fig1_a.png" width="800" />
# </p>
#
# <p>
#     <img src="imgs/Fig1_b.png" width="800" />
# </p>
#
# In more detail, MicroSplit performs the task of ***joint splitting and unsupervised denoising*** in fluorescence microscopy.
#
# From a technical perspective, given a noisy image with superimposed labeled structures $X$ (e.g., multiple fluorescently labeled structures imaged in the same channel), the goal is to predict multiple, **unmixed and denoised** images $C_1$, ..., $C_k$, each one corresponding to one of the $k$ different structures. Mathematically: $X = C_1 + C_2 + \dots + C_k + n$, where $n$ is the noise in $X$. 
#
# MicroSplit is trained with 2 main objectives (losses):
# - ***Supervised unmixing*** using target (noisy) unmixed images of the labeled structures.
# - ***Unsupervised denoising*** using a *Noise Model loss*. 
#
# MicroSplit's architecture is a slightly modified Variational Auto-Encoder (VAE).
# Specifically, it implements multiple latent spaces in a hierarchical manner. 
# For this reason the architecture is called Ladder VAE (LVAE).
# A distinctive feature of the model is the use of an additional trick called Lateral Contextualization (LC). 
# It consists in having additional inputs in the Encoder part which include larger field-of-views (FOVs) of the main superimposed input. 
# This enables the Neural Network to receive more long-range content than the one of a single input patch, hence allowing the extraction
# of global features which has shown to increase accuracy and consistency of unmixed predictions. 
#
# <p>
#     <img src="imgs/Fig2.png" width="800" />
# </p>
#
# ***NOTE***: you are now probably wondering how we get a larger FOVs for LC if the input is a full microscopy image. 
# Well... the reality is that in microscopy images are usually pretty large and GPUs are *always* too small (🥲). 
# Therefore, we usually work on image **patches** obtained by cropping parts of the image. 
# In this context, LC inputs are simply crops centered on the original one including a larger area of the image.

# %% [markdown] tags=[]
# ***References:***
# - VAE paper: [Kingma et al., Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
# - LVAE paper: [Sønderby et al, Ladder Variational Autoencoders](https://arxiv.org/abs/1602.02282)
# - MicroSplit paper: [Ashesh et al., Micro𝕊plit: Semantic Unmixing of Fluorescent Microscopy Data](https://www.nature.com/articles/s41592-026-03082-1)

# ***Additional resources:***
# - For more information about LC, please check this paper where we first introduced the idea: [μSplit: efficient image decomposition for microscopy data](https://openaccess.thecvf.com/content/ICCV2023/papers/Ashesh_uSplit_Image_Decomposition_for_Fluorescence_Microscopy_ICCV_2023_paper.pdf), which enabled the network to understand the global spatial context around the input patch.
# - To understand in detail how the joint denoising is performed please check this other work: [denoiSplit: a method for joint microscopy image splitting and unsupervised denoising](https://eccv.ecva.net/virtual/2024/poster/2538).
# P.S. Web is full of videos and blogposts explaining VAEs and LVAEs... just google it!

# %% [markdown] tags=[]
# <div class="alert alert-danger">
# Set your python kernel to <code>05_image_restoration</code>
# </div>

# %% [markdown] tags=["solution"]
# <div class="alert">
# Note for reviewers: dataset can be reduced to have faster training and inference!
# </div>

# %% tags=[]
from functools import partial
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from careamics.lightning import VAEModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from microsplit_reproducibility.configs.data.custom_dataset_2D import get_data_configs
from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config,
    get_optimizer_config,
    get_training_config,
    get_lr_scheduler_config,
)
from microsplit_reproducibility.configs.parameters._base import SplittingParameters
from microsplit_reproducibility.datasets import create_train_val_datasets
from microsplit_reproducibility.utils.callbacks import get_callbacks
from microsplit_reproducibility.utils.io import load_checkpoint_path
from microsplit_reproducibility.utils.utils import plot_input_patches, clean_ax
from microsplit_reproducibility.notebook_utils.custom_dataset_2D import (
    get_unnormalized_predictions,
    get_target,
    get_input,
    show_sampling,
    pick_random_patches_with_content,
)


from utils import (
    get_train_val_data,
    compute_metrics,
    show_metrics,
    full_frame_evaluation,
    load_pretrained_model,
    STRUCTURE_2_INDEX
)

# %matplotlib inline

# %% tags=[]
assert torch.cuda.is_available()
torch.set_float32_matmul_precision('medium')

# %% tags=[]
# TODO: update once loaded on Janelia servers
ROOT_DIR = Path("/group/jug/public_html/MicroSplit_MBL_2025")  # Path to the data folder


# %% [markdown] tags=[]
# # **Exercise 1**: Training MicroSplit

# %% [markdown] tags=[]
# ## 1.1. Data Preparation

# %% [markdown] tags=[]
# Since MicroSplit is trained in a supervised manner, we must feed:
# *(i)* input images containing the superimposed structures, and 
# *(ii)* target images/channels, each one showing one label structure separately. 
#
# Notice that, for simplicity, the mixed input image is here obtained synthetically by overlapping the other two channels (pixel-wise sum).
#
# In this exercise, we will use a dataset imaged at the *National Facility for Light Imaging at Human Technopole*.
#
# This dataset contains four labeled structures: 
# 1. Cell Nuclei,
# 1. Microtubules,
# 1. Nuclear Membrane,
# 1. Centromeres/Kinetocores.
#
# Additionally, this dataset offers acquisitions taken with different exposure times `(2, 20, 500 ms)`. 
# Hence, the data is available at various [signal-to-noise ratios](https://en.wikipedia.org/wiki/Signal-to-noise_ratio#:~:text=Signal%2Dto%2Dnoise%20ratio%20(,power%2C%20often%20expressed%20in%20decibels.)) (SNR). 
# Shorter exposure times only allows the collection of fewer photons, leading to higher *Poisson shot noise* and, therefore, a lower SNR.

# %% [markdown] tags=[]
# <div class="alert alert-info"><h4><b>Task 1.1.</b></h4>
#
# In the following, you will be asked to select:
# 1. The labeled structures to unmix;
# 2. The exposure time (and, thus, the SNR) of the input superimposed images.
#
# Observe that:
# - The more structures to unmix you pick, the more challenging the task becomes. A 2-structures unmixing is always easier than 3 or 4-structures unmixing.
# - The lower the SNR of the data you will choose to train $\mathrm{Micro}\mathbb{S}\mathrm{plit}$ with, the more challenging the task becomes and the more important will the unsupervised denoising feature of $\mathrm{Micro}\mathbb{S}\mathrm{plit}$ becomes.
#
# You can play with these parameters and check MicroSplit performance with different combinations.
# </div>

# %% tags=["task"]
# pick structures and exposure time
STRUCTURES = [..., ...] # choose among "Nuclei", "Microtubules", "NucMembranes", "Centromeres"
EXPOSURE_TIME = ... # in ms, choose among 2, 20, 500 (expressed in ms)

assert EXPOSURE_TIME in [2, 20, 500], "Exposure time must be one of [2, 20, 500] ms"
assert all([
    s in ["Nuclei", "Microtubules", "NucMembranes", "Centromeres"] for s in STRUCTURES
]), "Invalid structure selected. Choose among 'Nuclei', 'Microtubules', 'NucMembranes', 'Centromeres'."

# %% tags=["solution"]
# pick structures and exposure time
STRUCTURES = ["Nuclei", "Microtubules"] # choose among "Nuclei", "Microtubules", "NucMembranes", "Centromeres"
EXPOSURE_TIME = 500 # in ms, choose among 2, 20, 500 (expressed in ms)

assert EXPOSURE_TIME in [2, 20, 500], "Exposure time must be one of [2, 20, 500] ms"
assert isinstance(STRUCTURES, list), "Pass structures as a list"
assert all([
    s in ["Nuclei", "Microtubules", "NucMembranes", "Centromeres"] for s in STRUCTURES
]), "Invalid structure selected. Choose among 'Nuclei', 'Microtubules', 'NucMembranes', 'Centromeres'."
assert len(STRUCTURES) > 1, "Choose at leaser two structures"

# %% [markdown] tags=[]
# The following cell creates data configuration for training, validation and test sets. Each configurarion defines a set of parameters related to data loading, dataset creation and data processing.
#
# For more accessibility, configs are automatically generated by the `get_data_configs` wrapper function, which only requires to define:
# - `image_size` : `tuple[int]` -> the patch size used to train the model.
# - `num_channels` : `int` -> the number of target structures to unmix.

# %% tags=[]
train_data_config, val_data_config, test_data_config = get_data_configs(
    image_size=(64, 64),
    num_channels=len(STRUCTURES),
)

# %% [markdown] tags=[]
# Create the train, val, and test datasets

# %% tags=[]
datapath = ROOT_DIR / f"data/{EXPOSURE_TIME}ms"
load_data_func = partial(get_train_val_data, structures=STRUCTURES)

# %% tags=[]
# NOTE: here we are loading data from disk, creating synthetic inputs, generating patches... this might take a while
train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
    datapath=datapath,
    train_config=train_data_config,
    val_config=val_data_config,
    test_config=val_data_config,
    load_data_func=load_data_func,
)

# %% [markdown] tags=[]
# Create the train and val dataloaders. The test one is not used for training, hence it will be created later on during evaluation 

# %% tags=[]
train_dloader = DataLoader(
    train_dset,
    batch_size=32,
    num_workers=3,
    shuffle=True,
)
val_dloader = DataLoader(
    val_dset,
    batch_size=32,
    num_workers=3,
    shuffle=False,
)

# %% [markdown] tags=[]
# <div class="alert alert-info"><h4><b>Check some training data patches!</b></h4>
#
# ***Tip:*** the following functions shows a few samples of the prepared training data. In case you don't like what you see (empty or noisy patches), execute the cell again. Different randomly chosen patches will be shown!</div>

# %% tags=[]
plot_input_patches(dataset=train_dset, num_channels=len(STRUCTURES), num_samples=3, patch_size=64)

# %% [markdown] tags=[]
# <div class="alert alert-warning"><h4><b>Question 1.1.</b></h4>
#
# - Can you tell in which parts of the model the different patches shown below are used?
# - What do the input patches show? Why are there multiple inputs?
# - Why do we need targets? How do we use such targets? 

# %% [markdown] tags=["solution"]
# *Answers*
# - First columns contain, respectively, the superimposed input patch, and the additional input patches for Lateral Contextualization (LC). The later columns show, instead, the target unmixed patches.
# - Input patches represent the image obtained by superimposing (mixing) the signal coming from different labeled structures. The additional LC inputs are used to enhance the field of view and, hence, the semantic context processed by the network.
# - For this task we need unmixing targets as we are doing Supervised Learning.
# </div>

# %% [markdown] tags=[]
# <div class="alert alert-warning"><h4><b>Question 1.2.</b></h4>
#
# Below are 2 examples of superimposed labeled structures with the correspondent ground truths. 
# 1. Which one you think it's harder to unmix? Why?
# 2. What are, in your opinion, features of the input data that would make unmixing more difficult? 
#
# </div>

# %% [markdown] tags=["solution"]
# *Answers*
# 1. (b), because it shows more morphologically similar structures. MicroSplit is a content-aware method, i.e., it extracts semantic information regarding morphology, shape, brightness, etc., from the input data. Since structurally similar signal share many semantic features, the unmixing task becomes more challenging.
# 2. Semantic similarity between labeled structures, difference in brightness/intensity between labeled structures, colocalization, ...
# </div>

# %% [markdown] tags=[]
# <p>
#     <img src="imgs/question1.png" width="800" />
# </p>

# %% [markdown] tags=[]
# <div class="alert alert-success"><h2><b>Checkpoint 1: Data Preparation</b></h2>
# </div>
#
# <hr style="height:2px;">

# %% [markdown] tags=[]
# ## 1.2. Setup MicroSplit for training
# In this section, we create all the configs for the upcoming model initialization and training run. Configs allow to group all the affine parameters in the same place (architecture, training, loss, etc. etc.) and offer automated validation of the input parameters to prevent the user from inputting wrong combinations.
#
# Notice that MicroSplit is being implemented in CAREamics library, therefore the API is quite similar to the one you saw for Noise2Void. 

# %% [markdown] tags=[]
# <div class="alert alert-info"><h4><b>Task 1.2.</b></h4>
#
# In the following, we will initialize all the different configs that are needed to (i) instantiate a working MicroSplit model; (ii) train the model. 
#
# To facilitate the task, we define all the customizable parameters within the `get_microsplit_parameters` function. This returns the parameters for the current experiment that are later used to automatically define the single configs.
#
# Your task here is to:
# - Understand the meaning of these customizable parameters.
# - Play with some of them (only the ones indicated by us). 
#
# </div>

# %% [markdown] tags=[]
# Here's a break down of the customizable parameters:
#
# - `algorithm : str` -> the type of training algorithm to use. `denoisplit` does joint splitting and denoising, whereas `musplit` only does splitting.
# - `loss_type : str` -> the loss used to train the model. `denoisplit_musplit` practically means that the loss used is a combination of the loss used for `denoisplit` algorithm with some elements taken from `musplit` algorithm to improve splitting performance. 
# - `image_size : tuple[int]` -> the patch size used to train the model.
# - `noise_model_path : str` -> path to a folder containing the pre-trained noise models for the different channels.
# - `target_channels : int` -> the number of target structures to unmix (similar to `num_channels` seen before).
# - `multiscale_count : int` -> the total number of inputs (main + LC) to use. For instance, if set to `3`, it means main input + 2 LC inputs are used.
# - `lr : float` -> learning rate.
# - `num_epochs : int` -> the maximum number of epochs the model is trained for.
# - `lr_scheduler_patience : int` -> number of epochs to wait before decreasing the learning rate. Helps get away from local minima during training. Used by callbacks.
# - `earlystop_patientce : int` -> number of epochs to wait before applying earlystop. Earlystop consists in stopping training process before it reaches the desired number of epoche is a monitored metric (usually the aggregated validation loss) doesn't improve over some amount of epochs.

# %% [markdown] tags=[]
# Get pre-trained noise models

# %% tags=[]
NM_PATH = ROOT_DIR / f"noise_models/{EXPOSURE_TIME}ms"

paths_to_noise_models = [
    str(NM_PATH / f"noise_model_Ch{STRUCTURE_2_INDEX[structure]}.npz")
    for structure in STRUCTURES
]

# %% [markdown] tags=[]
# Set other parameters
# %% tags=["task"]
# setting up MicroSplit parametrization
experiment_params = SplittingParameters(
    algorithm="denoisplit",
    loss_type="denoisplit_musplit",
    img_size=(64, 64), # this should be consistent with the dataset
    target_channels=len(STRUCTURES),
    multiscale_count=3,
    lr=1e-3,
    num_epochs=..., # <- you can modify this
    lr_scheduler_patience=..., # <- you can modify this (note: if you want this to work, must be less than num_epochs)
    earlystop_patience=..., # <- you can modify this (note: if you want this to work, must be less than num_epochs)
    nm_paths=paths_to_noise_models,
).model_dump()

# add data stats for standardization
experiment_params["data_stats"] = data_stats

# %% tags=["solution"]
# setting up MicroSplit parametrization
experiment_params = SplittingParameters(
    algorithm="denoisplit",
    loss_type="denoisplit_musplit",
    img_size=(64, 64), # this should be consistent with the dataset
    target_channels=len(STRUCTURES),
    multiscale_count=3,
    lr=1e-3,
    num_epochs=100, # <- you can modify this
    lr_scheduler_patience=10, # <- you can modify this (note: if you want this to work, must be less than num_epochs)
    earlystop_patience=100, # <- you can modify this (note: if you want this to work, must be less than num_epochs)
    nm_paths=paths_to_noise_models,
).model_dump()

# add data stats for standardization
experiment_params["data_stats"] = data_stats

# %% tags=[]
# setting up training losses and model config (using default parameters)
loss_config = get_loss_config(**experiment_params)
model_config = get_model_config(**experiment_params)
gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(
    **experiment_params
)
training_config = get_training_config(**experiment_params)

# setting up learning rate scheduler and optimizer (using default parameters)
lr_scheduler_config = get_lr_scheduler_config(**experiment_params)
optimizer_config = get_optimizer_config(**experiment_params)

# finally, assemble the full set of experiment configurations...
experiment_config = create_algorithm_config(
    algorithm=experiment_params["algorithm"],
    loss_config=loss_config,
    model_config=model_config,
    gaussian_lik_config=gaussian_lik_config,
    nm_config=noise_model_config,
    nm_lik_config=nm_lik_config,
    lr_scheduler_config=lr_scheduler_config,
    optimizer_config=optimizer_config,
)

# %% tags=[]
model = VAEModule(algorithm_config=experiment_config)

# %% [markdown] tags=[]
# ## 1.3. Train MicroSplit model
#
# In this section we will train out MicroSplit model using `lightning`. We have manually set the time limit to 10 minutes. This limit can be modified with the `max_time` argument.

# %% tags=[]
# create the Trainer
trainer = Trainer(
    max_time="00:00:10:00", # this is roughly the time to train for 1.5 epochs
    max_epochs=training_config.num_epochs,
    accelerator="gpu",
    enable_progress_bar=True,
    callbacks=get_callbacks("./checkpoints/"),
    precision=training_config.precision,
    gradient_clip_val=training_config.gradient_clip_val,
    gradient_clip_algorithm=training_config.gradient_clip_algorithm,
    logger=TensorBoardLogger("tb_logs")
)

# %% tags=[]
# start the training
trainer.fit(
    model=model,
    train_dataloaders=train_dloader,
    val_dataloaders=val_dloader,
)

# %% [markdown] tags=[]
# **NOTE**: each training epoch should take approximately 7 minutes on our GPUs.
# After only 10 minutes of training, results will not be so good, but you will still be able to evaluate the model and see how it works.
# For reference, in our experiments we usually train MicroSplit for 50 or 100 epochs on similarly sized datasets,
# which takes approximately between 6 and 12 hours on a single GPU.

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h5><b>Task 1.3: Visualize losses and metrics using Tensorboard</b></h5>
#
# Open Tensorboard in VS Code to monitor training as you did it for the `01_CARE` and `02_Noise2Void` exercises!
# In this case, you have to open a terminal and run:
#
# ```
# conda activate 05_image_restoration
# tensorboard --logdir 04_MicroSplit/tb_logs/
# ```

# %% [markdown] tags=[]
# ## 1.4 Visualize predictions on validation data

# %% [markdown] tags=[]
# In order to check that the training process has been successful, we check MicroSplit predictions on the validation set.

# %% [markdown] tags=[]
# <div class="alert alert-warning"><h4><b>Question 1.4.</b></h4>
#
# A proper evaluation including prediction on mutliple images and computation of performance metrics will be performed later on the test data.
# Do you remember what are the limitations of evaluating a model's perfomance on the validation set, instead?
#
# </div>

# %% [markdown] tags=["solution"]
# **Answer**
# 
# The validation set can be used to: control the learning rate, decide when to stop training and tune the hyperparameters.
# Therefore, even though we did not adjust the model's parameters to minimize validation loss, the model is still technically fitted to the validation data.
# Why? Because we usually use validation loss and metrics to make decisions about the model (e.g., hyperparameter tuning, when to stop training, etc.).
# So our model would be biased towards performing well on the validation set.
# To properly test generalisation ability, we need to evaluate on data that was not used at all during training, and that data would be our test set.

# %% [markdown] tags=[]
# Before proceeding with the evaluation, let's focus once more on how MicroSplit works.
#
# As we mentioned, MicroSplit uses a modified version of the Ladder Variational Autoencoder (LVAE) similarly to other models you might have encountered during the course. 
# This architecture, given an input patch, enables the generation of multiple outputs. Technically, this happens by sampling multiple different *latent vectors* in the latent space. 
# In mathematical terms we say that "*MicroSplit is learning a full posterior of possible solutions*".
#
# This is a cool feature that makes our variational models pretty powerful and handy!!!
# Indeed, averaging multiple samples (predictions) generally allows to get smoother, more consistent predictions (in other terms, it somehow averages out potential "hallucinations" of the network). 
# Moreover, by computing the pixel-wise standard deviation over multiple samples (predictions) we can obtain a preliminary estimate of the (data) uncertainty in the model's predictions.
#
# In this framework, the parameter `mmse_count : (int)` determines the number of samples (predictions) generated for any given input patch. 
# A larger value allows to get smoother predictions, also limiting recurring issues such as *tiling artefacts*. However, it obviously increases the time and cost of the computation. 
# Generally, a value of `> 5` is enough to get decently smooth predicted frames. For reference, in our papers we often use values of 50 to get the best results. 

# %% tags=["task"]
MMSE_COUNT = ...
"""The number of MMSE samples to use for the splitting predictions."""

# %% tags=["solution"]
MMSE_COUNT = 5
"""The number of MMSE samples to use for the splitting predictions."""

# %% tags=[]
# Reduce the validation dataset to a single structure for quicker prediction
val_dset.reduce_data([0])

# Get patch predictions for the validation dataset + stitching into full images + de-normalization
stitched_predictions, _, _ = get_unnormalized_predictions(
    model,
    val_dset,
    data_key=val_dset._fpath.name,
    mmse_count=MMSE_COUNT,
    num_workers=3,
    grid_size=48,
    batch_size=32
)

# %% tags=[]
# get the target and input from the validation dataset for visualization purposes
tar = get_target(val_dset)
inp = get_input(val_dset).sum(-1)

# %% tags=[]
frame_idx = 0
assert frame_idx < len(stitched_predictions), f"Frame index {frame_idx} out of bounds. Max index is {len(stitched_predictions) - 1}."

full_frame_evaluation(stitched_predictions[frame_idx], tar[frame_idx], inp[frame_idx], same_scale=False)

# %% [markdown] tags=[]
# <div class="alert alert-success"><h2><b>Checkpoint 2: Model Training</b></h2>
# </div>
#
# <hr style="height:2px;">

# %% [markdown] tags=[]
# # **Exercise 2**: Evaluating MicroSplit performance
#
# So far, you have trained MicroSplit and had a first qualitative evaluation on the validation set. 
# However, at this point of the course you should be familiar with the idea that a proper evaluation should be carried out on a held-out test set,
# which has not been seen by the model during any part of the training process. In this section we perform the evaluation on the test set, 
# which will include a further qualitative inspection of predicted images and a quantitative evaluation using adequate metrics to measure models' performance
#
# Recall that for this task, on a standard GPU, we cannot feed the entire image to $\mathrm{Micro}\mathbb{S}\mathrm{plit}$. 
# Hence, we process smaller chunks of the full image that we so far called **patches**. 
# Usually, at training time these patches are obtained as random crops from the full input images, as random cropping works
# as a kind of ***data augmentation*** technique. However, at test time we want our predictions to be done on the full images.
# Hence, we need a more "organized" strategy to obtain the patches. An option is to divide the full frames into an ordered grid of patches.
# In our paper, we call this process ***tiling*** and we call the single crops ***tiles***, to differentiate them from the ones we use for training.
#
# A recurrent issue in ***tiled prediction*** is the possible presence of the so-called ***tiling artefacts***, which originate from inconsistencies and mismatches at the borders of neighboring tiles 
# (see (c) - No padding in the figure below). This problem can be alleviated by performing ***padding*** of the input tile, and later discarding the padded area when stitching the predictions. 
# The idea here is to introduce some overlap between neighboring tiles to have a smoother transition between them. Common padding strategies are:
# - ***Outer padding***: the patch (tile) size used for training (e.g., `(64, 64)`) is padded to a larger size. Then, the padded are is discarded during stitching.
# - ***Inner padding***: the patch (tile) size used for training (e.g., `(64, 64)`) is used as input for prediction. Then, only the inner part of it is kept during stitiching.
#
# In our work we use ***Inner padding*** as it preserves the same field of view the network has seen during training and empirically provides better performance on our task (see (b)).
#
# <p>
#     <img src="imgs/tiling.png" width="800" />
# </p>

# %% [markdown] tags=[]
# ## 2.1. Compute MicroSplit predictions on the test set

# %% [markdown] tags=[]
# <div class="alert alert-info"><h4><b>(Optional) Task 2.1.1: Load checkpoint</b></h4>
#
# In case you had any troubles while executing the notebook (disconnection, dead kernel, ...), you can avoid retraining MicroSplit from scratch and load, instead, some of your previous checkpoints.
#
# Similarly, if you are not satisfied with your trained model, you can try with the pre-trained one by us. However, we strongly suggest you first try with yours to identify any potential shortcoming, and then you resort back to ours to check how close you got to that.
#
# ***Note***: unfortunately we cannot provide pre-trained models for all the possible combinations of structures and exposures (if you're curious, there would be 33 combinations of such parameters 😌). Therefore, we provide one pre-trained model for each exposure time with 3 labeled structures to unmix (specifically, microtubules, nuclear membranes and centrosomes).
#
# So, run the cells in **Option A** to evaluate the model that you trained, or run the cells in **Option B** to evaluate a model that is pretrained.
#
# </div>

# %% [markdown] tags=[]
# ---
# #### **Option A**: load your previous checkpoints
#
# In the same subdirectory of the current `exercise.ipynb` notebook, you should see a `checkpoints` folder. This contains the checkpoints of your past training run.
#
# ***NOTE***: checkpoints are associated to a specific model configuration. Hence, if you changed any parameter in the configuration, you will not be able to load your previous checkpoints.
#
# ***NOTE***: if there are multiple checkpoints, our function will automatically pick the first found by listing the files in the folder.

# %% tags=[]
selected_ckpt = load_checkpoint_path("./checkpoints", best=True)
print("✅ Selected model checkpoint:", selected_ckpt)

# %% [markdown] tags=[]
# #### End of **Option A**
# ---

# %% [markdown] tags=[]
# ---
# #### **Option B**: load pre-trained checkpoints
#
# As we mentioned above, we only have a few pre-trained checkpoints available. For this reason, we will need to reinstantiate configs, datasets, and model to make sure they coincide with one of the pre-trained models.

# %% [markdown] tags=[]
# <div class="alert alert-info"><h5><b>Task: Pick a pre-trained model</b></h5>
#
# As we said, you can choose your desired exposure time. Structures will be set to `"Microtubules", "NucMembranes", "Centromeres"`.
#
# </div>
# %% tags=["task"]
EXPOSURE_TIME = ... # in ms, choose among 2, 20, 500 (expressed in ms)

assert EXPOSURE_TIME in [2, 20, 500], "Exposure time must be one of [2, 20, 500] ms"

# %% tags=["solution"]
EXPOSURE_TIME = 2 # in ms, choose among 2, 20, 500 (expressed in ms)

assert EXPOSURE_TIME in [2, 20, 500], "Exposure time must be one of [2, 20, 500] ms"

# %% [markdown] tags=[]
# Load checkpoint

# %% tags=[]
pretrained_ckpt_path = ROOT_DIR / f"ckpts/{EXPOSURE_TIME}ms"
selected_ckpt = load_checkpoint_path(str(pretrained_ckpt_path), best=True)
print("✅ Selected model checkpoint:", selected_ckpt)

# %% [markdown] tags=[]
# Reinstantiate configs, datasets, model

# %% tags=[]
train_data_config, val_data_config, test_data_config = get_data_configs(
    image_size=(64, 64),
    num_channels=3,
)

# %% tags=[]
datapath = ROOT_DIR / f"data/{EXPOSURE_TIME}ms"
load_data_func = partial(get_train_val_data, structures=["Microtubules", "NucMembranes", "Centromeres"])

train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
    datapath=datapath,
    train_config=train_data_config,
    val_config=val_data_config,
    test_config=val_data_config,
    load_data_func=load_data_func,
)

# %% tags=[]
# get noise models
NM_PATH = ROOT_DIR / f"noise_models/{EXPOSURE_TIME}ms"
paths_to_noise_models = [
    str(NM_PATH / f"noise_model_Ch{STRUCTURE_2_INDEX[structure]}.npz")
    for structure in ["Microtubules", "NucMembranes", "Centromeres"]
]

# setting up MicroSplit parametrization
experiment_params = SplittingParameters(
    algorithm="denoisplit",
    loss_type="denoisplit_musplit",
    img_size=(64, 64),
    target_channels=3,
    multiscale_count=3,
    predict_logvar="pixelwise",
    nm_paths=paths_to_noise_models,
).model_dump()

# add data stats for standardization
experiment_params["data_stats"] = data_stats

# %% tags=[]
# setting up training losses and model config (using default parameters)
loss_config = get_loss_config(**experiment_params)
model_config = get_model_config(**experiment_params)
gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(
    **experiment_params
)
training_config = get_training_config(**experiment_params)

# setting up learning rate scheduler and optimizer (using default parameters)
lr_scheduler_config = get_lr_scheduler_config(**experiment_params)
optimizer_config = get_optimizer_config(**experiment_params)

# finally, assemble the full set of experiment configurations...
experiment_config = create_algorithm_config(
    algorithm=experiment_params["algorithm"],
    loss_config=loss_config,
    model_config=model_config,
    gaussian_lik_config=gaussian_lik_config,
    nm_config=noise_model_config,
    nm_lik_config=nm_lik_config,
    lr_scheduler_config=lr_scheduler_config,
    optimizer_config=optimizer_config,
)

# %% tags=[]
model = VAEModule(algorithm_config=experiment_config)

# %% [markdown] tags=[]
# Now that checkpoints are loaded, we load these pre-trained weights into the model.

# %% tags=[]
load_pretrained_model(model, selected_ckpt)

# %% [markdown] tags=[]
# #### End of **Option B**
# ---

# %% [markdown] tags=[]
# <div class="alert alert-info"><h4><b>Task 2.1.2: Get test set predictions</b></h4>
#
# Here we reuse the `get_unnormalized_predictions` you saw before to get the unmixed predicted images for the training set.
# You will have to:
# - Set `MMSE_COUNT` parameter, being careful at finding an appropriate trade-off between prediction quality (remember the tiling artefacts we discussed above) and computation time.
# Given our time contraints, a reasonable range to try is `[2, 10]`.
# - Set `INNER_TILE_SIZE` parameter, trying different values for inner padding. 
# Also here notice that a smaller `INNER_TILE_SIZE` entails larger padding/overlap between neighboring patches and, hence, more predictions to be done. 
# A reasonable range to try is `[16, 64]`, where `64` means that no padding is done (recall, we used a patch size of `64` during training).
# </div>

# %% tags=["task"]
MMSE_COUNT = ...
"""The number of MMSE samples to use for the splitting predictions."""
INNER_TILE_SIZE = ...
"""The inner tile size considered for the predictions."""

# %% tags=["solution"]
MMSE_COUNT = 2
"""The number of MMSE samples to use for the splitting predictions."""
INNER_TILE_SIZE = 32
"""The inner tile size considered for the predictions."""

# %% tags=[]
stitched_predictions, _, stitched_stds = (
    get_unnormalized_predictions(
        model,
        test_dset,
        data_key=test_dset._fpath.name,
        mmse_count=MMSE_COUNT,
        grid_size=INNER_TILE_SIZE,
        num_workers=3,
        batch_size=32,
    )
)

# %% [markdown] tags=[]
# ***NOTE***: you might have seen that the function also returns `stitched_stds`. These are the pixel-wise standard deviations over the `MMSE_COUNT`-many samples for each image (yes, also these have been stitched back to images)!!

# %% [markdown] tags=[]
# <div class="alert alert-success"><h2><b>Checkpoint 3: Test set predictions</b></h2>
# </div>
#
# <hr style="height:2px;">

# %% [markdown] tags=[]
# ## 2.2. Qualitative evaluation of MicroSplit predictions
#
# In this section you will provided with tools to interactively inspect the predicted unmixed images from the test set to have a premliminary qualitative evaluation and spot potential issues.

# %% [markdown] tags=[]
# <div class="alert alert-info"><h4><b>Task 2.2: Look for defects in the obtained predictions</b></h4>
#
# Previously we discussed how noise, number of labeled structures, and morphological similarity between label structures can influence the complexity of the unmixing task. Depending on these factors, you might see some defects on your predicted unmixed images. In addition, we mentioned that tiled prediction can cause the so-called tiling artefacts.
#
# In this section, your task is to:
# 1. identify these defects (if any).
# 2. determine what is the likely source (e.g., tiling artefact, unmixing failure, ...).
#
# You will be provided with functions to visualize (i) full images, (ii) random smaller crops, (iii) custom crops.
# </div>

# %% tags=[]
# get the target and input from the test dataset for visualization purposes
tar = get_target(test_dset)
inp = get_input(test_dset).sum(-1)

# %% [markdown] tags=[]
# #### (i) Full image visualization

# %% tags=[]
frame_idx = 0 # Change this index to visualize different frames
assert frame_idx < len(stitched_predictions), f"Frame index {frame_idx} out of bounds. Max index is {len(stitched_predictions) - 1}."

full_frame_evaluation(stitched_predictions[frame_idx], tar[frame_idx], inp[frame_idx], same_scale=False)

# %% [markdown] tags=[]
# #### (ii) Random crops visualization
# %% tags=["task"]
# --- Insert here the crop size for visualization ---
img_sz = ...
# ---

rand_locations = pick_random_patches_with_content(tar, img_sz)

ncols = 1 + 2 * stitched_predictions.shape[-1]
nrows = min(len(rand_locations), 5)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))

for i, (h_start, w_start) in enumerate(rand_locations[:nrows]):
    ax[i, 0].imshow(inp[0, h_start : h_start + img_sz, w_start : w_start + img_sz])
    for j in range(ncols // 2):
        ax[i, 2 * j + 1].imshow(
            tar[0, h_start : h_start + img_sz, w_start : w_start + img_sz, j]
        )
        ax[i, 2 * j + 2].imshow(
            stitched_predictions[
                0, h_start : h_start + img_sz, w_start : w_start + img_sz, j
            ]
        )

ax[0, 0].set_title("Primary Input")
for i in range(ncols // 2):  # 2 channel splitting
    ax[0, 2 * i + 1].set_title(f"Target Channel {i+1}")
    ax[0, 2 * i + 2].set_title(f"Predicted Channel {i+1}")

# reduce the spacing between the subplots
plt.subplots_adjust(wspace=0.03, hspace=0.03)
clean_ax(ax)

# %% tags=["solution"]
# --- Insert here the crop size for visualization ---
img_sz = 128
# ---

rand_locations = pick_random_patches_with_content(tar, img_sz)

ncols = 1 + 2 * stitched_predictions.shape[-1]
nrows = min(len(rand_locations), 5)
nrows = max(nrows, 2)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))

for i, (h_start, w_start) in enumerate(rand_locations[:nrows]):
    ax[i, 0].imshow(inp[0, h_start : h_start + img_sz, w_start : w_start + img_sz])
    for j in range(ncols // 2):
        ax[i, 2 * j + 1].imshow(
            tar[0, h_start : h_start + img_sz, w_start : w_start + img_sz, j],
        )
        ax[i, 2 * j + 2].imshow(
            stitched_predictions[
                0, h_start : h_start + img_sz, w_start : w_start + img_sz, j
            ],
        )

ax[0, 0].set_title("Primary Input")
for i in range(ncols // 2):  # 2 channel splitting
    ax[0, 2 * i + 1].set_title(f"Target Channel {i+1}")
    ax[0, 2 * i + 2].set_title(f"Predicted Channel {i+1}")

# reduce the spacing between the subplots
plt.subplots_adjust(wspace=0.03, hspace=0.03)
clean_ax(ax)

# %% [markdown] tags=[]
# #### (iii) Custom crop visualization

# %% tags=["task"]
# --- Pick coordinates of upper-left corner and crop size ---
y_start = ...
x_start = ...
crop_size = ...
#--------------
assert y_start + crop_size <= stitched_predictions.shape[1], f"y_start + crop_size exceeds image height, which is {stitched_predictions.shape[1]}"
assert x_start + crop_size <= stitched_predictions.shape[2], f"x_start + crop_size exceeds image width, which is {stitched_predictions.shape[2]}"

ncols = 1 + stitched_predictions.shape[-1]
nrows = 2
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5), constrained_layout=True)
ax[0, 0].imshow(inp[0, y_start : y_start + crop_size, x_start : x_start + crop_size])
for i in range(ncols - 1):
    ax[0, i + 1].imshow(
        tar[0, y_start : y_start + crop_size, x_start : x_start + crop_size, i],
    )
    ax[1, i + 1].imshow(
        stitched_predictions[
            0, y_start : y_start + crop_size, x_start : x_start + crop_size, i
        ],
    )
    ax[0, i + 1].set_title(f"Channel {i+1}", fontsize=15)

# disable the axis for ax[1,0]
ax[1, 0].axis("off")
ax[0, 0].set_title("Input", fontsize=15)
# set y labels on the right for ax[0,2]
ax[0,ncols-1].yaxis.set_label_position("right")
ax[0,ncols-1].set_ylabel("Target", fontsize=15)

ax[1,ncols-1].yaxis.set_label_position("right")
ax[1,ncols-1].set_ylabel("Predicted", fontsize=15)

print("Here the crop you selected:")

# %% tags=["solution"]
# --- Pick coordinates of upper-left corner and crop size ---
y_start = 750
x_start = 750
crop_size = 512
#--------------
assert y_start + crop_size <= stitched_predictions.shape[1], f"y_start + crop_size exceeds image height, which is {stitched_predictions.shape[1]}"
assert x_start + crop_size <= stitched_predictions.shape[2], f"x_start + crop_size exceeds image width, which is {stitched_predictions.shape[2]}"

ncols = 1 + stitched_predictions.shape[-1]
nrows = 2
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5), constrained_layout=True)
ax[0, 0].imshow(inp[0, y_start : y_start + crop_size, x_start : x_start + crop_size])
for i in range(ncols - 1):
    ax[0, i + 1].imshow(
        tar[0, y_start : y_start + crop_size, x_start : x_start + crop_size, i],
    )
    ax[1, i + 1].imshow(
        stitched_predictions[
            0, y_start : y_start + crop_size, x_start : x_start + crop_size, i
        ],
    )
    ax[0, i + 1].set_title(f"Channel {i+1}", fontsize=15)

# disable the axis for ax[1,0]
ax[1, 0].axis("off")
ax[0, 0].set_title("Input", fontsize=15)
# set y labels on the right for ax[0,2]
ax[0,ncols-1].yaxis.set_label_position("right")
ax[0,ncols-1].set_ylabel("Target", fontsize=15)

ax[1,ncols-1].yaxis.set_label_position("right")
ax[1,ncols-1].set_ylabel("Predicted", fontsize=15)

print("Here the crop you selected:")

# %% [markdown] tags=[]
# <div class="alert alert-warning"><h4><b>Question 2.2.</b></h4>
#
# Can you come up with any idea about how to get rid of the current issues in the predictions? 
# Take into account the things we mentioned during the course so far...
#
# </div>

# %% [markdown] tags=[]
# <div class="alert alert-warning"><h4><b>Bonus Question</b></h4>
#
# In this and other exercises we spoke of "tiling artefacts". These are generally due to a mismatch in the predictions of adjacent tiles/patches. 
# In the context of CNN and, specifically, VAE-based models, can you think about reasons why we have such effect?
#
# *Hint1*: for CNN, think about how convolution works at the image borders... <br>
# *Hint2*: for VAE, reflect on the sampling happening in the latent space....

# %% [markdown] tags=["solution"]
# **Answer**
#
# Tiling artifacts in CNNs arise because predictions near tile borders are not made under the same conditions as predictions in the tile interior. 
# For same-size convolutions, the network pads the tile boundaries. 
# Therefore, output pixels close to the edge of a tile depend partly on artificial padding values instead of real image context. 
# When two neighboring tiles are processed independently, their border predictions may be based on different artificial contexts, so the stitched image can show seams or discontinuities.

# The severity depends on the model's receptive field: all pixels whose receptive field intersects a tile boundary are potentially affected. 
# This is why overlapping tiles and discarding border predictions helps. 
# The final stitched image should preferably keep only the central region of each tile, where the receptive field is fully supported by real image content.

# For VAE-based models there is an additional source of inconsistency. 
# The prediction is stochastic: each tile is reconstructed from samples drawn from a learned latent distribution. 
# Adjacent tiles are sampled independently, so they may choose slightly different plausible explanations, intensity levels, textures, or structure assignments. 
# Even if each tile is locally reasonable, the sampled predictions may not agree at tile boundaries. 
# Averaging multiple samples, i.e. increasing `mmse_count`, reduces this stochastic mismatch.

# </div>

# %% [markdown] tags=[]
# ## 2.3. Quantitative evaluation of MicroSplit predictions
#
# In this section you will perform a quantitative evaluation of MicroSplit unmixing performance using the provided function to compute metrics. In image restoration there are several commonly used metrics to quantitatively assess the goodness of a model's predictions. Clearly, different metrics focus on different aspects and provide different insights. Some metrics evaluate the ***pixel-wise similarity*** between images, while some other focus on higher-order features (e.g., brightness, contrast, ...) and, hence, we say they evaluate the ***perceptual similarity*** of images. Some commonly used metrics are:
# - ***Pixel-wise similarity***: `Peak Signal-to-Noise Ratio (PSNR)`, `Pearson's Correlation Coefficient`.
# - ***Perceptual similarity***: `Structural similarity index measure (SSIM)` with its multi-scale variant `(MS-SSIM)`, and our variant for microscopy `MicroSSIM` (paper: [link](https://arxiv.org/abs/2408.08747)) with its multi-scale variant `(MicroMS3IM)`, `Learned Perceptual Image Patch Similarity (LPIPS)``.

# %% [markdown] tags=[]
# <div class="alert alert-info"><h4><b>Task 2.3: Compute metrics</b></h4>
#
# Here, your task is to select appropriate metrics to use for the quantitative evaluation among the available ones.
#
# *Hint*: there are no absolutely good and bad metrics. All the metrics are useful! They key is to understand *what they are telling you*.
# </div>

# %% tags=[]
# Comment out the metrics you don't want to use
METRICS = [
    "PSNR",
    "Pearson",
    "SSIM",
    "MS-SSIM",
    "MicroSSIM",
    "MicroMS3IM",
    "LPIPS",
]

# %% [markdown] tags=[]
# **NOTE**: as ground truth reference for computing the metrics, we will use the high-SNR images obtained with long exposure (500ms).

# %% tags=[]
_, _, gt_test_dset, _ = create_train_val_datasets(
    datapath=ROOT_DIR / "data/500ms",
    train_config=train_data_config,
    val_config=val_data_config,
    test_config=val_data_config,
    load_data_func=load_data_func,
)
gt_target = gt_test_dset._data

# %% [markdown] tags=[]
# Metrics computation (NOTE: should you get any warnings, don't worry, computation is still ok!)

# %% tags=[]
metrics_dict = compute_metrics(gt_target, stitched_predictions, metrics=METRICS)
show_metrics(metrics_dict)

# %% [markdown] tags=[]
# <div class="alert alert-warning"><h4><b>Question 2.3.</b></h4>
#
# - Do you spot inconsistencies between your qualitative judgement and the computed metrics? Did you expect something different?
# - Which metrics are the most informative/interpretable?
#
# </div>

# %% [markdown] tags=[]
# <div class="alert alert-success"><h2><b>Checkpoint 4: Qualitative and Quantitative evaluation</b></h2>
# </div>
#
# <hr style="height:2px;">

# <div class="alert alert-block alert-info"><h3>Next: Bonus exercise</h3>
#
# The next exercise is a bonus exercise as we don't expect you to reach to that point in the given time.
# But, if you do and you want to keep going, you have the opportunity to work with [03_bonus_COSDD](../03_bonus_COSDD/exercise.ipynb).
# COSDD is also a denoiser trained using unpaired noisy images: however, differently from N2V, it can handle structured noise.
# Specifically, it can handle noise that is correlated along rows.
# Row-correlated noise is common in scanning-based imaging techniques like point-scanning confocal microscopy, an example is shown below.
# It can also be found when using sCMOS sensors.
# The practical trade-off with N2V is that COSDD takes much longer to train as it uses the same architecture as MicroSplit, a Ladder VAE.
#
# <img src="./../04_bonus_COSDD/resources/structured noise.png">
#