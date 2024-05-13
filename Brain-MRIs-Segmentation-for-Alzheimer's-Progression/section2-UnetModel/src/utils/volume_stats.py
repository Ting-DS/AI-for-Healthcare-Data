"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Dice3D. If you completed exercises in the lessons
    # you should already have it.
    intersection = np.sum(a*b)
    volumes = np.sum(a) + np.sum(b)

    if volumes == 0:
        return -1

    return 2.*float(intersection) / float(volumes)

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    # ^ as in don't call the function ?
    intersection = np.sum(a*b)
    volumes = np.sum(a) + np.sum(b)
    union = volumes - intersection
    
    if union == 0:
        return -1
    
    return float(intersection)/float(union)


# Sensitivity and specificity
# the lesser the sensitivity, the worse the under-segmentation
def sensitivity(a,b):
    tp = np.sum(b[a==b])
    fn = np.sum(b[a!=b])

    if fn+tp == 0:
        return -1

    return (tp)/(fn+tp)
    

# the lesser the specificity, the worse the over-segmentation
def specificity(a,b):
    # let's reverse the meaning of the values
    a, b = a*(-1)+1, b*(-1)+1
    return sensitivity(a, b)
#     tn = np.sum(b[a==b])
#     fp = np.sum(b[a!=b])

#     if tn+fp == 0:
#         return -1

#     return (tn)/(tn+fp)