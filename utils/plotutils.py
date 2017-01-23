import numpy as np
import matplotlib.pyplot as plt
import matplotlib

"""
Author: Dylan Payton taken from FeedbackLCA code
Pad data with ones for visualization
Outputs:
  padded version of input
Args:
  data: np.ndarray
"""

def pad_data(data):
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
    (1, 1), (1, 1))                       # add some space between filters
    + ((0, 0),) * (data.ndim - 3))        # don't pad the last dimension (if there is one)
    padded_data = np.pad(data, padding, mode="constant", constant_values=-1)
    # tile the filters into an image
    padded_data = padded_data.reshape((n, n) + padded_data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, padded_data.ndim + 1)))
    padded_data = padded_data.reshape((n * padded_data.shape[1], n * padded_data.shape[3]) + padded_data.shape[4:])
    return padded_data


"""
Author: Dylan Payton taken from FeedbackLCA code
Display input data as an image with reshaping
Outputs:
  fig: index for figure call
  sub_axis: index for subplot call
  axis_image: index for imshow call
Inpus:
  data: np.ndarray of shape (height, width) or (n, height, width)
  normalize: [bool] indicating whether the data should be streched (normalized)
    This is recommended for dictionary plotting.
  title: string for title of figure
  prev_fig: tuple containing (fig, sub_axis, axis_image) from previous
    display_data() call
  TODO: Allow for color weight vis
"""
def display_data_tiled(data, normalize=False, title="", prev_fig=None):
    if normalize:
        data = normalize_data(data)
    if len(data.shape) >= 3:
        data = pad_data(data)
    if prev_fig is None:
        plt.figure(figsize=(10,10))
        fig, sub_axis = plt.subplots(1)
        axis_image = sub_axis.imshow(data, cmap="Greys", interpolation="nearest")
        axis_image.set_clim(vmin=-1.0, vmax=1.0)
        # Turn off tick labels
        sub_axis.set_yticklabels([])
        sub_axis.set_xticklabels([])
        cbar = fig.colorbar(axis_image)
        sub_axis.tick_params(
         axis="both",
         bottom="off",
         top="off",
         left="off",
         right="off")
    else:
        (fig, sub_axis, axis_image) = prev_fig
        axis_image.set_data(data)
    fig.suptitle(title, y=1.05)
    #if prev_fig is None:
    #    fig.show()
    #else:
    #    fig.canvas.draw()
    return #(fig, sub_axis, axis_image)
