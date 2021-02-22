import numpy as np
from scipy.ndimage import morphology
import os
import SimpleITK as sitk



def MeanSurfaceDistance(maskA, maskM):
    """
    DESCRIPTION: Calculates mean surface distance (MSD) between the outer edges of two surfaces.
    For each point on the outer edge of the automatic segmentation the distance to the point closest on the outer edge
    of the manual segmentation is calculated and also vice verse; for each point on the outer edge of the manual
    segmentation the distance to the point closest on the outer edge of the automatic segmentation is calculated.
    The mean of all these distances is calculated and that is the MSD.
    ----------
    INPUTS:
    maskA = automatic segmentation
    maskB = manual segmentation
    -------
    OUTPUTS:
    MSD
    """
    input_1 = np.atleast_1d(maskA.astype(np.bool))
    input_2 = np.atleast_1d(maskM.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, 1)

    S = (input_1.astype('uint8') - (morphology.binary_erosion(input_1, conn).astype('uint8'))).astype('bool')
    Sprime = (input_2.astype('uint8') - (morphology.binary_erosion(input_2, conn).astype('uint8'))).astype('bool')

    # voxelsize die uit het artikel van Pluim komt
    sampling = [0.55, 0.55, 3]

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])
    msd = sds.mean()

    return msd
