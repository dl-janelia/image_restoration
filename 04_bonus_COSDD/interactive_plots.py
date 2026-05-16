import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interactive_output
import ipywidgets as widgets

from COSDD import utils


def find_dark_patch(low_snr):
    vmin = np.percentile(low_snr, 1)
    vmax = np.percentile(low_snr, 99)

    num_images = low_snr.shape[0]
    num_channels = low_snr.shape[1]
    max_height = low_snr.shape[2]
    max_width = low_snr.shape[3]

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
    autocorr_button = widgets.ToggleButton(description="Calculate autocorrelation", layout=widgets.Layout(width='200px'))


    def reset_toggle(*args):
        autocorr_button.value = False


    index_slider.observe(reset_toggle, "value")
    vertical_widget.observe(reset_toggle, "value")
    horizontal_widget.observe(reset_toggle, "value")


    def plot_crop(image_index, horizontal, vertical, plot_ac=False):
        left, right = horizontal[0], horizontal[1]
        top, bottom = vertical[1], vertical[0]
        top = max_height - top
        bottom = max_width - bottom
        crop = (image_index, 0, slice(top, bottom), slice(left, right))
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(low_snr[crop], vmin=vmin, vmax=vmax)
        if plot_ac:
            max_lag = min(min(25, bottom - top), min(25, right - left))
            noise_ac = utils.autocorrelation(low_snr[crop], max_lag=max_lag)
            ac = ax[1].imshow(noise_ac, cmap="seismic", vmin=-1, vmax=1)
            fig.colorbar(ac, fraction=0.045)
            ax[1].set_title("Autocorrelation plot")
            ax[1].set_xlabel("Horizontal lag")
            ax[1].set_ylabel("Vertical lag")
        else:
            ax[1].imshow(np.zeros_like(low_snr[crop]), cmap="seismic", vmin=-1, vmax=1)
            ax[1].axis("off")
        plt.show()


    interactive_output_widget = interactive_output(
        plot_crop,
        {
            "image_index": index_slider,
            "horizontal": horizontal_widget,
            "vertical": vertical_widget,
            "plot_ac": autocorr_button,
        },
    )

    index_and_ac = widgets.HBox([index_slider, autocorr_button])
    sliders = widgets.HBox([horizontal_widget, vertical_widget])
    slide_and_index = widgets.VBox([index_and_ac, sliders])

    layout = widgets.VBox([slide_and_index, interactive_output_widget])
    return layout


def plot_samples(test_data, samples):
    vmin = np.percentile(test_data, 1)
    vmax = np.percentile(test_data, 99)
    n_samples = samples.shape[1]

    num_images = test_data.shape[0]
    num_channels = test_data.shape[1]
    max_height = test_data.shape[2]
    max_width = test_data.shape[3]

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
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax[0, 0].imshow(test_data[image_index][crop], vmin=vmin, vmax=vmax)
        ax[0, 0].set_title("Input")
        for i in range(3):
            ax.flatten()[i + 1].imshow(
                samples[image_index][i][crop], vmin=vmin, vmax=vmax
            )
            ax.flatten()[i + 1].set_title(f"Sample {i+1}")
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

    return layout


def plot_mmse(test_data, MMSEs, samples):
    vmin = np.percentile(test_data, 1)
    vmax = np.percentile(test_data, 99)

    num_images = test_data.shape[0]
    num_channels = test_data.shape[1]
    max_height = test_data.shape[2]
    max_width = test_data.shape[3]

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
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(test_data[image_index][crop], vmin=vmin, vmax=vmax)
        ax[0].set_title("Input")
        ax[1].imshow(samples[image_index][0][crop], vmin=vmin, vmax=vmax)
        ax[1].set_title("Sample")
        ax[2].imshow(MMSEs[image_index][crop], vmin=vmin, vmax=vmax)
        ax[2].set_title("MMSE")

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
    return layout


def plot_direct(test_data, direct, MMSEs):
    vmin = np.percentile(test_data, 1)
    vmax = np.percentile(test_data, 99)

    num_images = test_data.shape[0]
    num_channels = test_data.shape[1]
    max_height = test_data.shape[2]
    max_width = test_data.shape[3]

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
        top = 1024 - top
        bottom = 1024 - bottom
        crop = (0, slice(top, bottom), slice(left, right))
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(test_data[image_index][crop], vmin=vmin, vmax=vmax)
        ax[0].set_title("Input")
        ax[1].imshow(direct[image_index][crop], vmin=vmin, vmax=vmax)
        ax[1].set_title("Direct")
        ax[2].imshow(MMSEs[image_index][crop], vmin=vmin, vmax=vmax)
        ax[2].set_title("MMSE")

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
    return layout
