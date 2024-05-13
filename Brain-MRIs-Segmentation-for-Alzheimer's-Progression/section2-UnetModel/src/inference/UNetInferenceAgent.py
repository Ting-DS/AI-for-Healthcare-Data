"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        volume_padded = med_reshape(volume, new_shape=(self.patch_size, self.patch_size, self.patch_size))
        prediction_mask = self.single_volume_inference(volume_padded)
        return prediction_mask

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        # <YOUR CODE HERE>
        for i in range(volume.shape[0]):
            test_slice = torch.from_numpy(volume[i,:,:].astype(np.single)/np.max(volume[i,:,:]))
            test_slice = test_slice.unsqueeze(0).unsqueeze(0).to(self.device)
            pred = self.model(test_slice)
#             print(pred.shape)  # [1, 3, 64, 64]
            
            cpu_pred = pred.cpu()
            # class 0 is no hippocampus and 1,2 are the two segments of hippocampi
            result = cpu_pred.detach().numpy()[0]  # does .data instead of .detach() work ?
            result = np.argmax(result, axis=0)  # as fast as torch.argmax and doesn't return tensor but np array
           
            slices.append(result)

        return np.stack(slices)
