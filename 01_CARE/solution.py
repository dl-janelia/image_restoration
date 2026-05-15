# %% tags=["solution", "task"]
# ruff: noqa: F811
# %% [markdown] tags=[]
# # Content-aware image restoration
#
# Fluorescence microscopy is constrained by the microscope's optics, fluorophore chemistry, and the sample's photon tolerance. 
# These constraints require balancing imaging speed, resolution, light exposure, and depth. 
# CARE demonstrates how Deep Learning can extend the range of biological phenomena observable by microscopy when any of these factor becomes limiting.
#
# **Reference**: Weigert, et al. "Content-aware image restoration: pushing the limits of fluorescence microscopy." Nature methods 15.12 (2018): 1090-1097. doi:[10.1038/s41592-018-0216-7](https://www.nature.com/articles/s41592-018-0216-7)
#

# %% [markdown] tags=[]
# ### CARE
#
# In this first exercise we will train a CARE model for a 2D denoising task. 
# CARE stands for Content-Aware image REstoration, and is a supervised method in which we use pairs of degraded and high quality images to train a particular task. 
# The original paper demonstrated improvement of image quality on a variety of tasks such as image restoration or resolution improvement. 
# Here, we will apply CARE to denoise images acquired at low laser power in order to recover the biological structures present in the data!
#
# <p align="center">
#     <img src="nb_data/img_intro.png" alt="Denoising task" class="center"> 
# </p>
#
# We'll use the UNet model that we built previously and use a different set of functions to train the model for restoration rather than segmentation.
#
#
# <div class="alert alert-block alert-success"><h3>Objectives</h3>
#     
# - Train a UNet on a new task!
# - Understand how to train CARE
#   
# </div>
#

# %% [markdown] tags=[]
# <div class="alert alert-danger">
#   Set your python kernel to <code>05_image_restoration</code>
# </div>


# %% tags=[]
import tifffile
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple
from torch.utils.data import Dataset, DataLoader
import torch.nn
import torch.optim
from torch import no_grad, cuda
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dlmbl_unet import UNet

# %matplotlib inline


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# <hr style="height:2px;">
#
# ## Part 1: Set-up the data
#
# CARE is a fully supervised algorithm, therefore we need image pairs (noisy & clean) for training. 
# In practice this is best achieved by acquiring each image twice, once with short exposure time or low laser power to obtain a noisy low-SNR (signal-to-noise ratio) image, and once with high SNR.
#
# Here, we will be using high SNR images of Human U2OS cells taken from the Broad Bioimage Benchmark Collection ([BBBC006v1](https://bbbc.broadinstitute.org/BBBC006)). 
# The low SNR images were created by synthetically adding strong read-out and shot noise, and applying pixel binning of 2x2, thus mimicking acquisitions at a very low light level.
#
# Since the image pairs were synthetically created in this example, they are already aligned perfectly. 
# Note that when working with real paired acquisitions, the low and high SNR images are not pixel-perfect aligned so they would often need to be co-registered before training a CARE model.
# 

# %% [markdown] tags=[]
# ### Split the dataset into training and validation
#

# %%
# Define the paths
root_path = Path("./../data")
root_path = root_path / "denoising-CARE_U2OS.unzip" / "data" / "U2OS"
assert root_path.exists(), f"Path {root_path} does not exist"

train_images_path = root_path / "train" / "low"
train_targets_path = root_path / "train" / "GT"
test_image_path = root_path / "test" / "low"
test_target_path = root_path / "test" / "GT"


image_files = list(Path(train_images_path).rglob("*.tif"))
target_files = list(Path(train_targets_path).rglob("*.tif"))
assert len(image_files) == len(
    target_files
), "Number of images and targets do not match"

print(f"Total size of train dataset: {len(image_files)}")

# %%
# Split the train data into train and validation
seed = 42
train_files_percentage = 0.8
np.random.seed(seed)
shuffled_indices = np.random.permutation(len(image_files))
image_files = np.array(image_files)[shuffled_indices]
target_files = np.array(target_files)[shuffled_indices]
assert all(
    [i.name == j.name for i, j in zip(image_files, target_files)]
), "Files do not match"

train_image_files = image_files[: int(train_files_percentage * len(image_files))]
train_target_files = target_files[: int(train_files_percentage * len(target_files))]
val_image_files = image_files[int(train_files_percentage * len(image_files)) :]
val_target_files = target_files[int(train_files_percentage * len(target_files)) :]
assert all(
    [i.name == j.name for i, j in zip(train_image_files, train_target_files)]
), "Train files do not match"
assert all(
    [i.name == j.name for i, j in zip(val_image_files, val_target_files)]
), "Val files do not match"

print(f"Train dataset size: {len(train_image_files)}")
print(f"Validation dataset size: {len(val_image_files)}")

# Read the test files
test_image_files = list(test_image_path.rglob("*.tif"))
test_target_files = list(test_target_path.rglob("*.tif"))
print(f"Number of test files: {len(test_image_files)}")


# %% [markdown] tags=[]
# ### Patching function
#
# In the majority of cases microscopy images are too large to be processed at once and need to be divided into smaller patches. 
# We will define a function that takes image and target arrays and extracts **random** (paired) patches from them.
#
# The method is a bit scary because accessing the whole patch coordinates requires some magical python expressions. 
#

# %% tags=[]
def create_patches(
    image_array: np.ndarray,
    target_array: np.ndarray,
    patch_size: Union[List[int], Tuple[int, ...]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create random patches from an array and a target.

    The method calculates how many non-overlapping patches the image can be divided into 
    (i.e., if we were dividing the image in a grid of patches) and then extracts an equal
    number of random patches.

    Important: the images should have an extra dimension before the spatial dimensions,
    i.e., they are expected to be of shape (N, H, W) for 2D images or (N, D, H, W)
    for 3D images, where N is the number of samples.
    """
    # random generator
    rng = np.random.default_rng()
    image_patches = []
    target_patches = []

    # iterate over the number of samples in the input array
    for s in range(image_array.shape[0]):
        # calculate the number of patches we can extract
        sample = image_array[s]
        target_sample = target_array[s]
        n_patches = np.ceil(np.prod(sample.shape) / np.prod(patch_size)).astype(int)

        # iterate over the number of patches
        for _ in range(n_patches):
            # get random coordinates for the patch and create the crop coordinates
            # NOTE: we sample here the top-left corner of the patch
            crop_coords = [
                rng.integers(0, sample.shape[i] - patch_size[i], endpoint=True)
                for i in range(len(patch_size))
            ]

            # extract patch from the data
            patch = (
                sample[
                    (
                        ...,
                        *[
                            slice(c, c + patch_size[i])
                            for i, c in enumerate(crop_coords)
                        ],
                    )
                ]
                .copy()
                .astype(np.float32)
            )

            # same for the target patch
            target_patch = (
                target_sample[
                    (
                        ...,
                        *[
                            slice(c, c + patch_size[i])
                            for i, c in enumerate(crop_coords)
                        ],
                    )
                ]
                .copy()
                .astype(np.float32)
            )

            # add the patch pair to the list
            image_patches.append(patch)
            target_patches.append(target_patch)

    # return stack of patches
    return np.stack(image_patches), np.stack(target_patches)

# %% [markdown] tags=[]
# ### Create patches
#
# To train the network, we will use patches of size 128x128. 
# We first need to load the data, stack it and then call our patching function.

# %% tags=[]
# Load images from files and stack them into arrays
train_images_array = np.stack([tifffile.imread(str(f)) for f in train_image_files])
train_targets_array = np.stack([tifffile.imread(str(f)) for f in train_target_files])
val_images_array = np.stack([tifffile.imread(str(f)) for f in val_image_files])
val_targets_array = np.stack([tifffile.imread(str(f)) for f in val_target_files])

test_images_array = np.stack([tifffile.imread(str(f)) for f in test_image_files])
test_targets_array = np.stack([tifffile.imread(str(f)) for f in test_target_files])


print(f"Train images array shape: {train_images_array.shape}")
print(f"Validation images array shape: {val_images_array.shape}")
print(f"Test array shape: {test_images_array.shape}")

# %% tags=[]
# Create patches
patch_size = (128, 128)

train_images_patches, train_targets_patches = create_patches(
    train_images_array, train_targets_array, patch_size
)
assert (
    train_images_patches.shape[0] == train_targets_patches.shape[0]
), "Number of patches do not match"

val_images_patches, val_targets_patches = create_patches(
    val_images_array, val_targets_array, patch_size
)
assert (
    val_images_patches.shape[0] == val_targets_patches.shape[0]
), "Number of patches do not match"

print(f"Train images patches shape: {train_images_patches.shape}")
print(f"Validation images patches shape: {val_images_patches.shape}")


# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning"><h3>Question: Patch size</h3>
#
# In the cell above we set a patch size and created random patches images.
# What should we consider when choosing the patch size?
#
#
# </div>

# %% [markdown] tags=["solution"]
# <div class="alert alert-block alert-warning"><h3>Answer: Patch size</h3>
#
# For most practical applications, when choosing the patch size, we should consider the following factors:
# - **Content**: The patch size should be large enough to capture the relevant context and features of the image (e.g., potentially we should be able to capture an entire object within it, like a cell or a nucleus for instance).
# - **Receptive Field**: The patch size should be compatible with the receptive field of the neural network. If the patch size is smaller than the receptive field, the network may not be able to learn effectively.
# - **Computational Resources**: Larger patches require more memory and computational power. We need to balance the patch size with the available resources.
# - **Model Architecture**: Some models may have specific requirements or limitations regarding the input size. E.g., if we have multiple downsampling layers, the patch size should be divisible by a certain factor.
#
# </div>


# %% [markdown] tags=[]
# ### Visualize training patches

# %% tags=[]
fig, ax = plt.subplots(3, 2, figsize=(15, 15))
ax[0, 0].imshow(train_images_patches[0], cmap="magma")
ax[0, 0].set_title("Train image")
ax[0, 1].imshow(train_targets_patches[0], cmap="magma")
ax[0, 1].set_title("Train target")
ax[1, 0].imshow(train_images_patches[1], cmap="magma")
ax[1, 0].set_title("Train image")
ax[1, 1].imshow(train_targets_patches[1], cmap="magma")
ax[1, 1].set_title("Train target")
ax[2, 0].imshow(train_images_patches[2], cmap="magma")
ax[2, 0].set_title("Train image")
ax[2, 1].imshow(train_targets_patches[2], cmap="magma")
ax[2, 1].set_title("Train target")
plt.tight_layout()

# %% [markdown] tags=[]
# ### Dataset class
#
# In modern deep learning libraries, the data is wrapped into a class called a `Dataset`. 
# Instances of that class are then used to extract the patches before feeding them to the network.
#
# Here, the class will be wrapped around our pre-computed stacks of patches. 
# Our `CAREDataset` class is built on top of the PyTorch `Dataset` class (we say it "inherits" from `Dataset`, the "parent" class). 
# That means that it has some function hidden from us that are defined in the PyTorch repository, but that we also need to implement specific pre-defined methods, such as `__len__` and `__getitem__`. 
# The advantage is that PyTorch knows what to do with a `Dataset` "child" class, since its behaviour is defined in the `Dataset` class, but we can do operations that are closely related to our own data in the method we implement.

# %% [markdown] tags=[]
# <div class="alert alert-block alert-warning"><h3>Question: Normalization</h3>
#
# In the following cell we calculate the mean and standard deviation of the input and target images so that we can normalize them.
# Why is normalization important? 
# Should we normalize the input and ground truth data the same way? 
#
# </div>

# %% [markdown] tags=["solution"]
# <div class="alert alert-block alert-warning"><h3>Answer: Normalization</h3>
# Normalization brings the data's values into a standardized range, making the default weight initialization appropriate and magnitude of gradients suitable for the default learning rate. 
# The target noise-free images have a much higher intensity than the noisy input images.
# They need to be normalized using their own statistics to bring them into the same range.
# </div>

# %%
# Calculate the mean and std of the train dataset
train_mean = train_images_array.mean()
train_std = train_images_array.std()
target_mean = train_targets_array.mean()
target_std = train_targets_array.std()
print(f"Train mean: {train_mean}, std: {train_std}")
print(f"Target mean: {target_mean}, std: {target_std}")

# %% [markdown] tags=[]
# The following functions will be used to normalize the data and perform data augmentation as it is loaded.

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 1: Normalization</h3>
#
# Define the normalization function. It should take an image, the mean and the standard deviation over the dataset and return the normalized image.
#
# </div>

# %% tags=["task"]
def normalize(
    image: np.ndarray,
    mean: float = 0.0,
    std: float = 1.0,
) -> np.ndarray:
    """
    Normalize an image with given mean and standard deviation.

    Parameters
    ----------
    image : np.ndarray
        Array containing single image or patch, 2D or 3D.
    mean : float, optional
        Mean value for normalization, by default 0.0.
    std : float, optional
        Standard deviation value for normalization, by default 1.0.

    Returns
    -------
    np.ndarray
        Normalized array.
    """
    return ... # YOUR CODE HERE

# %% tags=["solution"]
def normalize(
    image: np.ndarray,
    mean: float = 0.0,
    std: float = 1.0,
) -> np.ndarray:
    """
    Normalize an image with given mean and standard deviation.

    Parameters
    ----------
    image : np.ndarray
        Array containing single image or patch, 2D or 3D.
    mean : float, optional
        Mean value for normalization, by default 0.0.
    std : float, optional
        Standard deviation value for normalization, by default 1.0.

    Returns
    -------
    np.ndarray
        Normalized array.
    """
    return (image - mean) / std


# %% tags=[]
def _flip_and_rotate(
    image: np.ndarray, rotate_state: int, flip_state: int
) -> np.ndarray:
    """
    Apply the given number of 90 degrees rotations and flip to an array.

    Parameters
    ----------
    image : np.ndarray
        Array containing single image or patch, 2D or 3D.
    rotate_state : int
        Number of 90 degree rotations to apply.
    flip_state : int
        0 or 1, whether to flip the array or not.

    Returns
    -------
    np.ndarray
        Flipped and rotated array.
    """
    rotated = np.rot90(image, k=rotate_state, axes=(-2, -1))
    flipped = np.flip(rotated, axis=-1) if flip_state == 1 else rotated
    return flipped.copy()


def augment_batch(
    patch: np.ndarray,
    target: np.ndarray,
    seed: int | None = None,
) -> Tuple[np.ndarray, ...]:
    """
    Apply augmentation function to patches and masks.

    Parameters
    ----------
    patch : np.ndarray
        Array containing single image or patch, 2D or 3D with masked pixels.
    original_image : np.ndarray
        Array containing original image or patch, 2D or 3D.
    mask : np.ndarray
        Array containing only masked pixels, 2D or 3D.
    seed : int, optional
        Seed for random number generator, controls the rotation and flipping.

    Returns
    -------
    Tuple[np.ndarray, ...]
        Tuple of augmented arrays.
    """
    if seed is not None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()
    rotate_state = rng.integers(0, 4)
    flip_state = rng.integers(0, 2)
    return (
        _flip_and_rotate(patch, rotate_state, flip_state),
        _flip_and_rotate(target, rotate_state, flip_state),
    )

# %% [markdown] tags=[]
# ### Defining the Dataset
#
# Here we're defining the basic pytorch dataset class that will be used to load the data. 
# This class will be used to load the data and apply the normalization and augmentation functions to the data as it is loaded.
#


# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 2: Dataset</h3>
#
# Complete the `__len__` and `__getitem__` methods of `CAREDataset` class below.
#
# *Hint* : You should use the augmentation and normalization functions defined above.
# Check the description of class attributes defined in the `__init__` method to understand what each of them is.
# </div>

# %% tags=["task"]
# Define a Dataset
class CAREDataset(Dataset): # CAREDataset inherits from the PyTorch Dataset class
    def __init__(
        self, image_data: np.ndarray, target_data: np.ndarray, apply_augmentations: bool = False
    ):
        """Constructor.
        
        Parameters
        ----------
        image_data : np.ndarray
            Array containing all input images or patches, 2D (N, H, W) or 3D (N, D, H, W).
        target_data : np.ndarray
            Array containing all target images or patches, 2D (N, H, W) or 3D (N, D, H, W).
        apply_augmentations : bool, optional
            Whether to apply augmentations to the patches, by default False.
        """
        # data
        self.image_data = image_data
        self.target_data = target_data
        
        # data statistics
        self.image_data_mean = self.image_data.mean()
        self.image_data_std = self.image_data.std()
        self.target_data_mean = self.target_data.mean()
        self.target_data_std = self.target_data.std()
        
        # whether to apply augmentations
        self.patch_augment = apply_augmentations

    def __len__(self):
        """Return the total number of patches.

        This method is called when applying `len(...)` to an instance of our class
        """
        return ... # YOUR CODE HERE --> define the total number of patches

    def __getitem__(self, index: int):
        """Return a single pair of patches."""

        # get input noisy patch
        patch = ... # YOUR CODE HERE

        # get target clean patch
        target = ... # YOUR CODE HERE

        # Apply transforms
        if self.patch_augment:
            patch, target = ... # YOUR CODE HERE

        # Normalize the patch
        patch = ... # YOUR CODE HERE
        target = ... # YOUR CODE HERE

        return (
            patch[np.newaxis].astype(np.float32),
            target[np.newaxis].astype(np.float32)
        )

# %% tags=["solution"]
# Define a Dataset
class CAREDataset(Dataset): # CAREDataset inherits from the PyTorch Dataset class
    def __init__(
        self, image_data: np.ndarray, target_data: np.ndarray, apply_augmentations=False
    ):
        """
        Constructor.
        
        Parameters
        ----------
        image_data : np.ndarray
            Array containing all input images or patches, 2D (N, H, W) or 3D (N, D, H, W).
        target_data : np.ndarray
            Array containing all target images or patches, 2D (N, H, W) or 3D (N, D, H, W).
        apply_augmentations : bool, optional
            Whether to apply augmentations to the patches, by default False.
        """
        # data
        self.image_data = image_data
        self.target_data = target_data
        
        # data statistics
        self.image_data_mean = self.image_data.mean()
        self.image_data_std = self.image_data.std()
        self.target_data_mean = self.target_data.mean()
        self.target_data_std = self.target_data.std()
        
        # whether to apply augmentations
        self.patch_augment = apply_augmentations

    def __len__(self):
        """Return the total number of patches.

        This method is called when applying `len(...)` to an instance of our class
        """
        return self.image_data.shape[0]

    def __getitem__(self, index: int):
        """Return a single pair of patches."""

        # get input noisy patch
        patch = self.image_data[index]

        # get target clean patch
        target = self.target_data[index]

        # Apply transforms
        if self.patch_augment:
            patch, target = augment_batch(patch=patch, target=target)

        # Normalize the patch
        patch = normalize(patch, self.image_data_mean, self.image_data_std)
        target = normalize(target, self.target_data_mean, self.target_data_std)

        return (
            patch[np.newaxis].astype(np.float32),
            target[np.newaxis].astype(np.float32)
        )


# %% tags=[]
# test the dataset
train_dataset = CAREDataset(
    image_data=train_images_patches, target_data=train_targets_patches
)
val_dataset = CAREDataset(
    image_data=val_images_patches, target_data=val_targets_patches
)

# what is the dataset length?
assert len(train_dataset) == train_images_patches.shape[0], "Dataset length is wrong"

# check the normalization
assert train_dataset[42][0].max() <= 10, "Patch isn't normalized properly"
assert train_dataset[42][1].max() <= 10, "Target patch isn't normalized properly"

# check the get_item function
assert train_dataset[42][0].shape == (1, *patch_size), "Patch size is wrong"
assert train_dataset[42][1].shape == (1, *patch_size), "Target patch size is wrong"
assert train_dataset[42][0].dtype == np.float32, "Patch dtype is wrong"
assert train_dataset[42][1].dtype == np.float32, "Target patch dtype is wrong"


# %% [markdown] tags=[]
# The training and validation data are stored as an instance of a `Dataset`. 
# This describes how each image should be loaded.
# Now we will prepare them to be fed into the model with a `DataLoader`.
#
# This will use the Dataset to load individual images and organise them into batches.
# The Dataloader will shuffle the data at the start of each epoch, outputting different random batches.

# %% tags=[]
# Instantiate the dataset and create a DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success"><h1>Checkpoint 1: Data</h1>
#
# In this section, we prepared paired training data. 
# The steps were:
# 1) Loading the images.
# 2) Cropping them into patches.
# 3) Checking the patches visually.
# 4) Creating an instance of a pytorch Dataset and DataLoader.
#
# You'll see a similar preparation procedure followed for most deep learning vision tasks.
#
# Next, we'll use this data to train a denoising model.
# </div>
#
# <hr style="height:2px;">
#

# %% [markdown] tags=[]
# ## Part 2: Training the model
#
# Image restoration task is very similar to the segmentation task we have done in a previous exercise. 
# The main difference is that instead of predicting a mask, we want to predict a clean image from a noisy input image.
# Therefore, we can use the same UNet model and just need to adapt a few things.
#
# %% [markdown] tags=[]
# ![image](nb_data/carenet.png)
# %% [markdown] tags=[]
# ### Instantiate the model
#
# We'll be using the model from the previous exercise, so we need to load the relevant module.

# %% tags=[]
# Load the model
model = UNet(depth=2, in_channels=1, out_channels=1)
# NOTE: 1 grayscale image in, 1 grayscale image out

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 3: Loss function</h3>
#
# CARE trains image to image (output vs. ground truth, i.e., noisy vs. clean), therefore we need a different loss function compared to the segmentation task (image to mask). 
# For example, we may want to somehow measure the pixel-wise difference in intensity between the output and the ground truth.
# Can you think of a suitable loss function?
#
# *hint: look in the `torch.nn` module of PyTorch ([link](https://pytorch.org/docs/stable/nn.html#loss-functions)).*
#
# </div>

# %% tags=["task"]
loss = ... #### YOUR CODE HERE ####

# %% tags=["solution"]
loss = torch.nn.MSELoss()

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 4: Optimizer</h3>
#
# Similarly, define the optimizer. No need to be too inventive here!
#
# *hint* : look in the `torch.optim` module of PyTorch ([link](https://pytorch.org/docs/stable/optim.html)).
# Make sure to define all the parameters required for the optimizer (e.g., learning rate).
# </div>

# %% tags=["task"]
optimizer = ... #### YOUR CODE HERE ####

# %% tags=["solution"]
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4
)

# %% [markdown] tags=[]
# ### Training
#
# Here we will train a CARE model using classes and functions you defined in the previous tasks.
# We're using the same training loop as in the semantic segmentation exercise.
#

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 5: Launch Tensorboard</h3>
#
# We'll monitor the training of all models in 05_image_restoration using Tensorboard.
# This is a program that plots the training and validation loss of networks as they train,
# and can also show input/output image pairs.
#
# Follow these steps to launch Tensorboard:
#
# 1) Start training by running the cell below.
# 2) Open a terminal and run:
#
# ```bash
# conda activate 05_image_restoration
# tensorboard --logdir 01_CARE/runs/
# ```
#
# 3) Open [http://localhost:6006](http://localhost:6006) in your browser (in VSCode a window with the link will pop up).
# </div>

# %% [markdown] tags=[]
# In tensorboard, click the SCALARS tab to see the training and validation loss curves. 
# At the end of each epoch, refresh Tensorboard using the button in the top right to see the latest loss.
#
# Click the IMAGES tab to see the noisy inputs, denoised outputs and clean targets.
# These are updated at the end of each epoch too.

# %% tags=[]
# Training loop
n_epochs = 5
device = "cuda" if cuda.is_available() else "cpu"
model.to(device)

# tensorboard
tb_logger = SummaryWriter("runs/Unet"+datetime.now().strftime('%d%H-%M%S'))
def log_image(image, tag, logger, step):
    normalised_image = image.cpu().numpy()
    normalised_image = normalised_image - np.percentile(normalised_image, 1)
    normalised_image = normalised_image / np.percentile(normalised_image, 99)
    normalised_image = np.clip(normalised_image, 0, 1)
    logger.add_images(tag=tag, img_tensor=normalised_image, global_step=step)


train_losses = []
val_losses = []

for epoch in range(n_epochs):
    model.train()
    for i, (image_batch, target_batch) in enumerate(train_dataloader):
        batch = image_batch.to(device)
        target = target_batch.to(device)

        optimizer.zero_grad()
        output = model(batch)
        train_loss = loss(output, target)
        train_loss.backward()
        optimizer.step()

        if i % 20 == 0 or i == len(train_dataloader) - 1:
            print(f"Epoch: {epoch} - Batch: {i}/{len(train_dataloader)} - Loss: {train_loss.item()}")
            tb_logger.add_scalar(tag="train_loss", scalar_value=train_loss, global_step=epoch * len(train_dataloader) + i)

    model.eval()

    with no_grad():
        val_loss = 0
        for i, (batch, target) in enumerate(val_dataloader):
            batch = batch.to(device)
            target = target.to(device)

            output = model(batch)
            val_loss = loss(output, target)

        # log tensorboard
        step = epoch * len(train_dataloader)
        tb_logger.add_scalar(tag="val_loss", scalar_value=val_loss, global_step=step)

        # we always log the last validation images
        log_image(batch, tag="val_input", logger=tb_logger, step=step)
        log_image(target, tag="val_target", logger=tb_logger, step=step)
        log_image(output, tag="val_prediction", logger=tb_logger, step=step)

        print(f"Validation loss: {val_loss.item()}")

    # Save the losses for plotting
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())


# %% [markdown] tags=[]
# ### Plot the loss

# %% tags=[]
# Plot training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses)
plt.plot(val_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train loss", "Validation loss"])

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success"><h1>Checkpoint 2: Training</h1>
#
# In this section, we created and trained a UNet for denoising.
# We:
# 1) Instantiated the model with random weights.
# 2) Chose a loss function to compare the output image to the ground truth clean image.
# 3) Chose an optimizer to minimize that loss function.
# 4) Trained the model with this optimizer.
# 5) Examined the training and validation loss curves to see how well our model trained.
#
# Next, we'll load a test set of noisy images and see how well our model denoises them.
# </div>
#
# <hr style="height:2px;">
#

# %% [markdown] tags=[]
# ## Part 3: Predicting on the test dataset
#

# %%
# Define the dataset for the test data
test_dataset = CAREDataset(
    image_data=test_images_array, target_data=test_targets_array
)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 6: Denormalization</h3>
#
# CARE is an image to image model. If we feed it normalized images and use normalized targets for training, it will output normalized images.
# Therefore, we can map the model output back to the original intensity range by reverting the normalization operation, i.e., **denormalizing**.
# Define the denormalization function. 
# It should take a normalized image (e.g., the model output), the mean and the standard deviation over the dataset and return the denormalized image.
#
# *hint* : You just need to invert the normalization formula you defined above!
# </div>

# %% tags=["task"]
def denormalize(
    image: np.ndarray,
    mean: float = 0.0,
    std: float = 1.0,
) -> np.ndarray:
    """
    Denormalize an image with given mean and standard deviation.

    Parameters
    ----------
    image : np.ndarray
        Array containing single image or patch, 2D or 3D.
    mean : float, optional
        Mean value for normalization, by default 0.0.
    std : float, optional
        Standard deviation value for normalization, by default 1.0.

    Returns
    -------
    np.ndarray
        Denormalized array.
    """
    return ... # YOUR CODE HERE

# %% tags=["solution"]
def denormalize(
    image: np.ndarray,
    mean: float = 0.0,
    std: float = 1.0,
) -> np.ndarray:
    """
    Denormalize an image with given mean and standard deviation.

    Parameters
    ----------
    image : np.ndarray
        Array containing single image or patch, 2D or 3D.
    mean : float, optional
        Mean value for normalization, by default 0.0.
    std : float, optional
        Standard deviation value for normalization, by default 1.0.

    Returns
    -------
    np.ndarray
        Denormalized array.
    """
    return image * std + mean


# %% [markdown]
# <div class="alert alert-block alert-info"><h3>Task 7: Predict using the correct mean/std</h3>
#
# In Part 1 we normalized the inputs and the targets before feeding them into the model. 
# This means that the model will output normalized clean images. 
# However, we'd like them to be on the same scale as the real clean images.
#
# Recall the variables storing the dataset statistics we used to normalize the data in Part 1, and use them denormalize the output of the model.
# Should you use the mean and std of the input images or the target images?
#
# </div>

# %% tags=["task"]
# Define the prediction loop
predictions = []

model.eval()
with no_grad():
    for i, (image_batch, target_batch) in enumerate(test_dataloader):
        image_batch = image_batch.to(device)
        target_batch = target_batch.to(device)
        prediction = model(image_batch).cpu().numpy()
        
        # Denormalize the prediction
        prediction = ... # YOUR CODE HERE

        # Save the predictions for visualization
        predictions.append(prediction)

# %% tags=["solution"]
# Define the prediction loop
predictions = []

model.eval()
with no_grad():
    for i, (image_batch, target_batch) in enumerate(test_dataloader):
        image_batch = image_batch.to(device)
        target_batch = target_batch.to(device)
        prediction = model(image_batch).cpu().numpy()
        
        # Denormalize the prediction
        prediction = denormalize(prediction, target_mean, target_std)

        # Save the predictions for visualization
        predictions.append(prediction)

# %% [markdown] tags=[]
# ### Visualize the predictions
# %%
fig, ax = plt.subplots(3, 2, figsize=(15, 15))
ax[0, 0].imshow(test_images_array[0].squeeze(), cmap="magma")
ax[0, 0].set_title("Test image")
ax[0, 1].imshow(predictions[0][0].squeeze(), cmap="magma")
ax[0, 1].set_title("Prediction")
ax[1, 0].imshow(test_images_array[1].squeeze(), cmap="magma")
ax[1, 0].set_title("Test image")
ax[1, 1].imshow(predictions[1][0].squeeze(), cmap="magma")
ax[1, 1].set_title("Prediction")
ax[2, 0].imshow(test_images_array[2].squeeze(), cmap="magma")
ax[2, 0].set_title("Test image")
ax[2, 1].imshow(predictions[2][0].squeeze(), cmap="magma")
ax[2, 1].set_title("Prediction")
plt.tight_layout()

# %% [markdown] tags=[]
# <div class="alert alert-block alert-success"><h1>Checkpoint 3: Predicting</h1>
#
# In this section, we evaluated the performance of our denoiser.
# We:
# 1) Created a CAREDataset and Dataloader for a prediction loop.
# 2) Ran a prediction loop on the test data.
# 3) Examined the outputs.
#
# This notebook has shown how matched pairs of noisy and clean images can train a UNet to denoise.
# </div>

# %% [markdown] tags=[]
# <div class="alert alert-block alert-info"><h3>Task 8: Choose your next exercise</h3>
#
# TODO: update, in case we make N2V mandatory and remove COSDD
# You are free to choose which deep learning-based image restoration method you want to learn about next.
# To learn more about denoising, you can choose from [02_Noise2Void](../02_Noise2Void/exercise.ipynb) or [03_COSDD](../03_COSDD/exercise.ipynb).
# Or, to learn about computational unmixing, try [04_DenoiSplit](../04_DenoiSplit/exercise.ipynb).
#
# Notice that exercises have different levels of difficulty.
# Choose one that matches your confidence level (or be brave and try something harder)!
#
# (EASY) [02_Noise2Void](../02_Noise2Void/exercise.ipynb) is a denoiser that is trained directly on (unpaired) noisy images in a self-supervised fashion. Meaning that, unlike CARE, we don't need any examples of clean images.
# It's also relatively quick to train.
# But there's a catch.
# It relies on the assumption that the noise is unstructured.
# Unstructured noise is uncorrelated over pixels, so has no streaky or line artifacts.
# An example is shown below.
#
# <img src="./../02_Noise2Void/imgs/unstructured noise.png">
#
# (HARD) [03_COSDD](../03_COSDD/exercise.ipynb) is also a denoiser trained using unpaired noisy images, but it can handle a specific form of structure.
# That structure is row correlation.
# Row-correlated noise is common in scanning-based imaging techniques like point-scanning confocal microscopy, an example is shown below.
# It can also be found when using sCMOS sensors.
# The practical trade-off with N2V is that COSDD takes much longer to train.
#
# <img src="./../03_COSDD/resources/structured noise.png">
#
# (HARD) [04_MicroSplit](../04_MicroSplit/exercise.ipynb) is a computational multiplexing technique.
# It uses deep learning to separate multiple superimposed cellular structures within a single fluorescent image channel, turning one fluorescent channel into multiple ones (up to 4 in our work).
# Imaging multiple cellular structures in a single fluorescent channel effectively increases the available photon budget, which can be reallocated to achieve faster imaging, higher signal-to-noise ratios, or the imaging of additional structures. 
# An example of splitting is shown below.
# 
# <img src="./../04_MicroSplit/imgs/Fig1_b.png">
#
# </div>

# %%
