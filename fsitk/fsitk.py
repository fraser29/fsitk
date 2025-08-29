"""
@author Fraser Callaghan

SITK exmaples, helpers and FC pipelines
Examples from: https://simpleitk.readthedocs.io/en/master/Examples/index.html


"""

import os
import numpy as np
import vtk
import SimpleITK as sitk
from ngawari import vtkfilters
from ngawari import fIO
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider




def __generateRndTempFile(tempdir='/tmp'):
    # DEPRECIATED
    A, Z = np.array(["A", "Z"]).view("int32")
    LEN = 20
    tempstr = str(np.random.randint(low=A, high=Z, size=LEN, dtype="int32").view(f"U{LEN}")[0])
    return os.path.join(tempdir, tempstr+'.vtk')


def __vtsToVTI(fileName, arrayName):
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(fileName)
    reader.Update()
    vv = reader.GetOutput()
    vtkfilters.setArrayAsScalars(vv, arrayName)
    ii = vtkfilters.vtsToVti(vv)#, True)
    return ii


def setVTIArrayTo16int(ii, arrayName):
    vtkfilters.addNpArray(ii, vtkfilters.getScalarsAsNumpy(ii).astype(np.int16), arrayName, SET_SCALAR=True)
def setVTIArrayTo16uint(ii, arrayName):
    vtkfilters.addNpArray(ii, vtkfilters.getScalarsAsNumpy(ii).astype(np.uint16), arrayName, SET_SCALAR=True)

def sitk_image_fromArray(arr3D, origin, spacing):
    img_sitk = sitk.GetImageFromArray(arr3D)
    img_sitk.SetOrigin(origin)
    img_sitk.SetSpacing(spacing)
    return img_sitk

def sitkHelper_VTKToITKImage(fileName, arrayName=None):
    if type(fileName) == str:
        if os.path.splitext(fileName)[1] == '.vts':
            imgVTK = __vtsToVTI(fileName, arrayName)
        elif fileName.endswith('mha'):
            iiITK = sitk.ReadImage(fileName, sitk.sitkFloat32)
            return iiITK
        else:
            imgVTK = fIO.readVTKFile(fileName)
    else:
        imgVTK = fileName
    if arrayName is not None:
        imgA = vtkfilters.getArrayAsNumpy(imgVTK, arrayName)
    else:
        imgA = vtkfilters.getScalarsAsNumpy(imgVTK)
    dims = list(imgVTK.GetDimensions())
    if np.ndim(imgA) > 1:
        imgA = np.reshape(imgA, dims+[3], 'F')
        imgA = np.transpose(imgA, (2, 1, 0, 3))
    else:
        imgA = np.reshape(imgA, imgVTK.GetDimensions(), 'F')
        imgA = np.transpose(imgA, (2, 1, 0))

    oo = imgVTK.GetPoint(0)
    imgSITK = sitk_image_fromArray(imgA, oo, imgVTK.GetSpacing())
    imgSITK_ = sitk.Cast(imgSITK, sitk.sitkFloat32)
    return imgSITK_

def sitkHelper_ITKToVTKImage(imObj, arrayName='PixelData'):
    """

    :param imObj:
    :param arrayName: default PixelData
    :return:
    """
    ii = vtk.vtkImageData()
    ii.SetSpacing(imObj.GetSpacing())
    ii.SetOrigin(imObj.GetOrigin())
    k,j,i = imObj.GetSize()
    ii.SetDimensions([k,j,i])
    arr = sitk.GetArrayFromImage(imObj)
    if np.ndim(arr) == 4:
        arr = np.transpose(arr, (2, 1, 0, 3))
        arr = np.reshape(arr, (np.prod(arr.shape[:3]), arr.shape[-1]), 'F')
    else:
        arr = np.transpose(arr, (2, 1, 0))
        arr = np.ndarray.flatten(arr, 'F')
    vtkfilters.addNpArray(ii, arr, arrayName, SET_SCALAR=True)
    return ii

def sitkHelper_NumpyToITKImage(numpyA, origin=(0,0,0), spacing=(1,1,1), frmt=None, TO_TRANSPOSE=True):
    """
    Take numpy array and return sitk_image
    :param numpyA: 2D or 3D array. If 3D assumes [cols x rows x slices] and thus will transpose to [slices x rows x cols]
    :param origin: [0,0,0]
    :param spacing: [1,1,1] resolution
    :param frmt:
    :param TO_TRANSPOSE: [True] set to false if already in form [slices x rows x cols]
    :return:
    """
    if frmt is not None:
        numpyA = numpyA.astype(frmt)
    if TO_TRANSPOSE:
        numpyA = np.transpose(numpyA, (2, 1, 0))
    img_sitk = sitk.GetImageFromArray(numpyA)
    img_sitk.SetOrigin(origin)
    img_sitk.SetSpacing(spacing)
    return img_sitk

def getSliceAsNumpy(img, xSlice=None, ySlice=None, zSlice=None):
    if xSlice is not None:
        slice = sitk.GetArrayViewFromImage(img)[:, :, xSlice]
    elif ySlice is not None:
        slice = sitk.GetArrayViewFromImage(img)[:, ySlice, :]
    elif zSlice is not None:
        slice = sitk.GetArrayViewFromImage(img)[zSlice, :, :]
    else:
        raise ValueError('Must give a slice number (x, y, or z) to show')
    return slice


def _strOrImgToImage(ff_or_ImObj):
    if type(ff_or_ImObj) == str:
        try:
            return sitk.ReadImage(ff_or_ImObj)
        except RuntimeError:
            return sitkHelper_VTKToITKImage(ff_or_ImObj)
    else:
        try:
            if ff_or_ImObj.IsA('vtkImageData'):
                return sitkHelper_VTKToITKImage(ff_or_ImObj)
        except AttributeError:
            pass
        return ff_or_ImObj


def writeVTI_to_mha(ff_or_Img, fileOut, arrayName=None):
    iiITK = sitkHelper_VTKToITKImage(ff_or_Img, arrayName=arrayName)
    sitk.WriteImage(iiITK, fileOut)


def _showRegistrationResult(fixed, moving, outTx, txt='RESULT'):
    if (not "SITK_NOSHOW" in os.environ):
        out = transformMovingImage(fixed, moving, outTx)
        simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
        simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
        cimg = sitk.Compose(simg1, simg2, simg1//2.+simg2//2.)
        sitk.Show(cimg, txt, debugOn=True)

def filterMedian(imObj, rad=3):
    mF = sitk.MedianImageFilter()
    mF.SetRadius(rad)
    return mF.Execute(imObj)

'''
# This better:
# dnNL[:, :, k1] = denoise_nl_means(iA, h=20.15 * sigmaEst, patch_size=9, patch_distance=5)
def filterDenoise(imObj):
    mDN = sitk.PatchBasedDenoisingImageFilter() This was running all cpus at 100 for > 30 mins
    return mDN.Execute(imObj)
'''

def transformMovingImage(fixedImg, movingImg, transform):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixedImg)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    out = resampler.Execute(movingImg)
    return out


def registrationRigid(fixedImage_file_or_obj, movingImage_file_or_obj): #, max_translation, max_rotation=10):
    #
    fixed = _strOrImgToImage(fixedImage_file_or_obj)
    moving = _strOrImgToImage(movingImage_file_or_obj)
    #
    transform = sitk.Euler3DTransform()
    transform.SetTranslation((0, 0, 0))  # Initialize translation
    transform.SetRotation(0, 0, 0)  # Initialize rotation
    #
    registration_method = sitk.ImageRegistrationMethod()
    # registration_method.SetMetricAsMeanSquares()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetInitialTransform(transform)
    final_transform = registration_method.Execute(fixed, moving)
    #
    # # Get the translation and rotation from the final transform
    # translation = final_transform.GetTranslation()
    # rotation = final_transform.GetRotation()
    # # Clamp translation and rotation
    # clamped_translation = (
    #     max(-max_translation[0], min(max_translation[0], translation[0])),
    #     max(-max_translation[1], min(max_translation[1], translation[1])),
    #     max(-max_translation[2], min(max_translation[2], translation[2]))
    # )
    # clamped_rotation = (
    #     max(-max_rotation, min(max_rotation, rotation[0])),
    #     max(-max_rotation, min(max_rotation, rotation[1])),
    #     max(-max_rotation, min(max_rotation, rotation[2]))
    # )
    # # Set the clamped translation and rotation back to the transform
    # transform.SetTranslation(clamped_translation)
    # transform.SetRotation(clamped_rotation)

    resampled_image = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0)
    return final_transform, resampled_image



def registrationExhaustive(fixedImage_file_or_obj, movingImage_file_or_obj, transformOutputFile=None, DISPLAY_RESULT=False):
    """
    Perform exhaustive image registration - From Examples
    :param fixedImage_file_or_obj:
    :param movingImage_file_or_obj:
    :param transformOutputFile:
    :param DISPLAY_RESULT: Set true to display the result (not working)
    :return:
    """
    def command_iteration(method):
        if (method.GetOptimizerIteration() == 0):
            print("Scales: ", method.GetOptimizerScales())
        print("{0:3} = {1:7.5f} : {2}".format(method.GetOptimizerIteration(),
                                              method.GetMetricValue(),
                                              method.GetOptimizerPosition()))


    fixed = _strOrImgToImage(fixedImage_file_or_obj)
    moving = _strOrImgToImage(movingImage_file_or_obj)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    sample_per_axis = 12
    if fixed.GetDimension() == 2:
        tx = sitk.Euler2DTransform()
        # Set the number of samples (radius) in each dimension, with a
        # default step size of 1.0
        R.SetOptimizerAsExhaustive([sample_per_axis//2, 0, 0])
        # Utilize the scale to set the step size for each dimension
        R.SetOptimizerScales([2.0*np.pi/sample_per_axis, 1.0, 1.0])
    elif fixed.GetDimension() == 3:
        tx = sitk.Euler3DTransform()
        R.SetOptimizerAsExhaustive([sample_per_axis//2, sample_per_axis//2, sample_per_axis//4, 0, 0, 0])
        R.SetOptimizerScales([2.0*np.pi/sample_per_axis, 2.0*np.pi/sample_per_axis, 2.0*np.pi/sample_per_axis, 1.0, 1.0, 1.0])

    # Initialize the transform with a translation and the center of
    # rotation from the moments of intensity.
    tx = sitk.CenteredTransformInitializer(fixed, moving, tx)
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda:command_iteration(R))
    outTx = R.Execute(fixed, moving)

    print("-------")
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))

    if transformOutputFile is not None:
        sitk.WriteTransform(outTx, transformOutputFile)
    if DISPLAY_RESULT:
        _showRegistrationResult(fixed, moving, outTx, "ImageRegistrationExhaustive Composition")
    return outTx, transformMovingImage(fixed, moving, outTx)


def registrationDemons(fixedImage_file_or_obj, movingImage_file_or_obj, transformOutputFile=None,
                       DISPLAY_RESULT=False,
                       maxIterations=50):
    """

    :param fixedImage_file_or_obj:
    :param movingImage_file_or_obj:
    :param transformOutputFile:
    :param DISPLAY_RESULT:
    :param maxIterations:
    :return:
    """
    def command_iteration(rfilter):
        print("{0:3} = {1:10.5f}".format(rfilter.GetElapsedIterations(),
                                         rfilter.GetMetric()))

    fixed = _strOrImgToImage(fixedImage_file_or_obj)
    moving = _strOrImgToImage(movingImage_file_or_obj)

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    moving = matcher.Execute(moving, fixed)

    # The basic Demons Registration Filter
    # Note there is a whole family of Demons Registration algorithms included in SimpleITK
    demonsRegFilter = sitk.DemonsRegistrationFilter()
    demonsRegFilter.SetNumberOfIterations(maxIterations)
    # Standard deviation for Gaussian smoothing of displacement field
    demonsRegFilter.SetStandardDeviations(1.0)
    demonsRegFilter.AddCommand(sitk.sitkIterationEvent, lambda:command_iteration(demonsRegFilter))
    displacementField = demonsRegFilter.Execute(fixed, moving)

    outTx = sitk.DisplacementFieldTransform(displacementField)

    print("Number Of Iterations: {0}".format(demonsRegFilter.GetElapsedIterations()))
    print(" RMS: {0}".format(demonsRegFilter.GetRMSChange()))


    if transformOutputFile is not None:
        sitk.WriteTransform(outTx, transformOutputFile)
    if DISPLAY_RESULT:
        _showRegistrationResult(fixed, moving, outTx, "Demons Composition")
    del(matcher, demonsRegFilter, displacementField)
    return outTx, transformMovingImage(fixed, moving, outTx)


def registrationDiffeomorphicDemon(fixedImage_file_or_obj, movingImage_file_or_obj, displacementFieldOutputFile=None,
                                   maxIterations=50,
                                   blurVal=None,
                                   subsample=None,
                                   initTransform=None, RMSError=0.00001,
                                   maskImage_file_or_obj=None):
    """

    :param fixedImage_file_or_obj:
    :param movingImage_file_or_obj:
    :param displacementFieldOutputFile:
    :param maxIterations:
    :param blurVal: gaussian blur (physical coords)
    :param subsample: for speed
    :param initTransform: initial initTransform to use
    :param RMSError: default 1e-4
    :param maskImage_file_or_obj: mask for moving image
    :return:sitk.DisplacementFieldTransform , transformedImage
    """
    def command_iteration(rfilter):
        print("{0:3} = {1:10.5f}".format(rfilter.GetElapsedIterations(),
                                         rfilter.GetMetric()))

    fixed = _strOrImgToImage(fixedImage_file_or_obj)
    moving = _strOrImgToImage(movingImage_file_or_obj)

    fixed = sitk.RescaleIntensity(fixed)
    moving = sitk.RescaleIntensity(moving)

    if blurVal is not None:
        fixed = sitk.SmoothingRecursiveGaussian(fixed, blurVal)
        moving = sitk.SmoothingRecursiveGaussian(moving, blurVal)

    if subsample is not None:
        shrinkF = sitk.ShrinkImageFilter()
        shrinkF.SetShrinkFactor(subsample)
        moving = shrinkF.Execute(moving)
        if moving.GetSize() != fixed.GetSize(): # this takes care of case that fixed is updated to previous moving.
            shrinkF = sitk.ShrinkImageFilter()
            shrinkF.SetShrinkFactor(subsample)
            fixed = shrinkF.Execute(fixed)

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(100)
    matcher.ThresholdAtMeanIntensityOn()

    if maskImage_file_or_obj is not None:
        mask = _strOrImgToImage(maskImage_file_or_obj)
        moving = sitk.Mask(moving, mask)

    moving = matcher.Execute(moving, fixed)

    demonsRegFilter = sitk.DiffeomorphicDemonsRegistrationFilter()

    demonsRegFilter.SetNumberOfIterations(maxIterations)
    # Standard deviation for Gaussian smoothing of displacement field
    demonsRegFilter.SetSmoothDisplacementField(True)
    demonsRegFilter.SetStandardDeviations(1.5)
    demonsRegFilter.SetMaximumRMSError(RMSError)
    if initTransform is not None:
        initDisplacementField = sitk.TransformToDisplacementField(initTransform,
                                                                       sitk.sitkVectorFloat64,
                                                                       fixed.GetSize(),
                                                                       fixed.GetOrigin(),
                                                                       fixed.GetSpacing(),
                                                                       fixed.GetDirection())
    else:
        initDisplacementField = sitk.Image(fixed.GetWidth(),
                                                fixed.GetHeight(),
                                                fixed.GetDepth(),
                                                sitk.sitkVectorFloat64)
        initDisplacementField.CopyInformation(fixed)

    # demonsRegFilter.AddCommand(sitk.sitkIterationEvent, lambda:command_iteration(demonsRegFilter))
    displacementField = demonsRegFilter.Execute(fixed, moving, initDisplacementField)
    outTx = sitk.DisplacementFieldTransform(displacementField)
    print("Number Of Iterations: {0}".format(demonsRegFilter.GetElapsedIterations()))
    print(" RMS: {0}".format(demonsRegFilter.GetRMSChange()))
    if displacementFieldOutputFile is not None:
        sitk.WriteImage(displacementField, displacementFieldOutputFile)
    return outTx, transformMovingImage(fixed, moving, outTx)


def registrationBSpline(fixedImage_file_or_obj,
                        movingImage_file_or_obj,
                        transformOutputFile=None,
                        DISPLAY_RESULT=False,
                        maxIterations=100):
    """
    ## VERY LONG AND DOESN'T SEEM TO WORK WELL ON IMAGE VOLS.
    BSpline2 example
    :param fixedImage_file_or_obj:
    :param movingImage_file_or_obj:
    :param transformOutputFile: default not written
    :param DISPLAY_RESULT: set Tru to display result (not working - unless set to make png)
    :param maxIterations:
    :return:
    """
    def command_iteration(method):
        print("{0:3} = {1:10.5f}".format(method.GetOptimizerIteration(),
                                         method.GetMetricValue()))
        print("\t#: ", len(method.GetOptimizerPosition()))

    def command_multi_iteration(method):
        print("--------- Resolution Changing ---------")


    fixed = _strOrImgToImage(fixedImage_file_or_obj)
    moving = _strOrImgToImage(movingImage_file_or_obj)

    transformDomainMeshSize = [10]*moving.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed,
                                          transformDomainMeshSize)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(50)
    R.SetOptimizerAsGradientDescentLineSearch(5.0, maxIterations,
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetShrinkFactorsPerLevel([6, 2, 1])
    R.SetSmoothingSigmasPerLevel([6, 2, 1])
    R.AddCommand(sitk.sitkIterationEvent, lambda:command_iteration(R))
    R.AddCommand(sitk.sitkMultiResolutionIterationEvent, lambda:command_multi_iteration(R))

    outTx = R.Execute(fixed, moving)

    print("-------")
    print(outTx)
    print("Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
    print(" Iteration: {0}".format(R.GetOptimizerIteration()))
    print(" Metric value: {0}".format(R.GetMetricValue()))

    if transformOutputFile is not None:
        sitk.WriteTransform(outTx, transformOutputFile)
    if DISPLAY_RESULT:
        _showRegistrationResult(fixed, moving, outTx, "Diffeomorphic Demons Composition")
    del(R, tx)
    return outTx, transformMovingImage(fixed, moving, outTx)


def resgistrationRigid(fixedImage_file_or_obj,
                        movingImage_file_or_obj,):

    fixed = _strOrImgToImage(fixedImage_file_or_obj)
    moving = _strOrImgToImage(movingImage_file_or_obj)

    # Initial alignment of the two volumes
    transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    # Multi-resolution rigid registration using Mutual Information
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInitialTransform(transform)
    outTx = registration_method.Execute(fixed, moving)

    return outTx, transformMovingImage(fixed, moving, outTx)



## ---------------- https://github.com/SimpleITK/SimpleITK/issues/532 --------------------------------------------------
def smooth_and_resample(image, shrink_factor, smoothing_sigma):
    """
    Args:
        image: The image we want to resample.
        shrink_factor: A number greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigma: Sigma for Gaussian smoothing, this is in physical (image spacing) units, not pixels.
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigma and shrink factor.
    """
    smoothed_image = sitk.SmoothingRecursiveGaussian(image, smoothing_sigma)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz/float(shrink_factor)+0.5) for sz in original_size]
    new_spacing = [((original_sz-1)*original_spc)/(new_sz-1)
                   for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]
    return sitk.Resample(smoothed_image, new_size, sitk.Transform(),
                         sitk.sitkLinear, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0.0,
                         image.GetPixelID())


def multiscale_demons(registration_algorithm,
                      fixed_image, moving_image, initial_transform=None,
                      shrink_factors=None, smoothing_sigmas=None):
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.
    Args:
        registration_algorithm: Any registration algorithm that has an Execute(fixed_image, moving_image, displacement_field_image)
                                method.
        fixed_image: Resulting transformation maps points from this image's spatial domain to the moving image spatial domain.
        moving_image: Resulting transformation maps points from the fixed_image's spatial domain to this image's spatial domain.
        initial_transform: Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors: Shrink factors relative to the original image's size.
        smoothing_sigmas: Amount of smoothing which is done prior to resmapling the image using the given shrink factor. These
                          are in physical (image spacing) units.
    Returns:
        SimpleITK.DisplacementFieldTransform
    """
    # Create image pyramid.
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))

    # Create initial displacement field at lowest resolution.
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by the Demons filters.
    if initial_transform:
        initial_displacement_field = sitk.TransformToDisplacementField(initial_transform,
                                                                       sitk.sitkVectorFloat64,
                                                                       fixed_images[-1].GetSize(),
                                                                       fixed_images[-1].GetOrigin(),
                                                                       fixed_images[-1].GetSpacing(),
                                                                       fixed_images[-1].GetDirection())
    else:
        initial_displacement_field = sitk.Image(fixed_images[-1].GetWidth(),
                                                fixed_images[-1].GetHeight(),
                                                fixed_images[-1].GetDepth(),
                                                sitk.sitkVectorFloat64)
        initial_displacement_field.CopyInformation(fixed_images[-1])

    # Run the registration.
    initial_displacement_field = registration_algorithm.Execute(fixed_images[-1],
                                                                moving_images[-1],
                                                                initial_displacement_field)
    # Start at the top of the pyramid and work our way down.
    for f_image, m_image in reversed(list(zip(fixed_images[0:-1], moving_images[0:-1]))):
        initial_displacement_field = sitk.Resample(initial_displacement_field, f_image)
        initial_displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)

    return sitk.DisplacementFieldTransform(initial_displacement_field)

def invertTransform(sitkTransform):
    dispField = sitkTransform.GetDisplacementField()
    npimg = sitk.GetArrayFromImage(dispField)
    invertedField = sitk.GetImageFromArray(-1.0*npimg)
    invertedField.SetOrigin(dispField.GetOrigin())
    invertedField.SetSpacing(dispField.GetSpacing())
    invertedField.SetDirection(dispField.GetDirection())
    sitkTransform.SetDisplacementField(invertedField)
    return sitkTransform
### ====================================================================================================================
### ====================================================================================================================

def transformPolyData(pp, tfx):
    pp2 = vtkfilters.copyData(pp)
    nPts = pp2.GetNumberOfPoints()
    pts = pp2.GetPoints()
    for k9 in range(nPts):
        x,y,z = pp.GetPoints().GetPoint(k9)
        newX, newY, newZ = tfx.TransformPoint([x,y,z])
        pts.SetPoint(k9, [newX, newY, newZ])
    pp2.Modified()
    return pp2

def deformArrayA_to_ArrayB(AFixed, AMoving, maxIter=500, blurVal=None,
                       subSampleVal=None, RMSError=0.00001, fileOut=None):
    AFixed = AFixed - np.mean(AFixed)
    AFixed = AFixed / np.std(AFixed)
    AM = np.mean(AMoving)
    AMoving = AMoving - AM
    AS = np.std(AMoving)
    AMoving = AMoving / AS
    imFixed = sitk_image_fromArray(np.transpose(AFixed, (2, 1, 0)), [0.0,0.0,0.0], [1.0,1.0,1.0])
    imMoving = sitk_image_fromArray(np.transpose(AMoving, (2, 1, 0)), [0.0,0.0,0.0], [1.0,1.0,1.0])
    tx = None
    tx, img = registrationDiffeomorphicDemon(imFixed, imMoving,
                                             maxIterations=maxIter,
                                             blurVal=blurVal,
                                             subsample=subSampleVal,
                                             initTransform=tx, RMSError=RMSError)
    if fileOut is not None:
        sitk.WriteImage(img, fileOut)
    ii = sitkHelper_ITKToVTKImage(img)
    ## RENAME ARRAYS
    AOut = vtkfilters.getScalarsAsNumpy(ii) * AS
    AOut = AOut + AM
    vtkfilters.addNpArray(ii, AOut.astype(np.uint16), 'PixelData', SET_SCALAR=True)
    return ii, tx

def deformVTIA_to_VTIB(iiFixed, iiMoving, arrayNameFixed, arrayNameMoving, maxIter=500, blurVal=None,
                       subSampleVal=None, RMSError=0.00001, fileOut=None, iiMask=None):
    fixed = sitkHelper_VTKToITKImage(iiFixed, arrayName=arrayNameFixed)
    tx = None
    moving = sitkHelper_VTKToITKImage(iiMoving, arrayName=arrayNameMoving)
    tx, img = registrationDiffeomorphicDemon(fixed, moving,
                                             maxIterations=maxIter,
                                             blurVal=blurVal,
                                             subsample=subSampleVal,
                                             initTransform=tx, RMSError=RMSError, maskImage_file_or_obj=iiMask)
    if fileOut is not None:
        sitk.WriteImage(img, fileOut)
    ii = sitkHelper_ITKToVTKImage(img)
    ## RENAME ARRAYS
    vtkfilters.addNpArray(ii, vtkfilters.getScalarsAsNumpy(ii).astype(np.uint16), arrayNameFixed, SET_SCALAR=True)
    return ii, tx


def deformSurf2Surf(surfFixed, surfMoving, res, DEBUG=False):
    aName = 'PixelData'
    nDilate = 1 # ?? this good ??
    nMed = 3 # ?? this good ??
    ## BUILD IMAGES
    imFixed = vtkfilters.surf2ImageBW(surfFixed, aName, res, nDilate=nDilate, nMed=nMed)
    imMoving = vtkfilters.surf2ImageBW(surfMoving, aName, res, nDilate=nDilate, nMed=nMed)
    if DEBUG:
        print(fIO.writeVTKFile(imFixed, '/home/fraser/temp/temp.vti'))
    # ---------------------------------------
    ## TRANSFORM X TO ATLAS
    if DEBUG:
        lmX_at_A = vtkfilters.iterativeClosestPointsMorphing(surfMoving, surfFixed, 0.001, RIGID=False)[-1]
        print(fIO.writeVTKFile(lmX_at_A, '/home/fraser/temp/temp.vtp'))
    TXA_move2fix = vtkfilters.iterativeClosestPointsTransform(surfMoving, surfFixed, 0.0001, RIGID=False)
    # Got transform, now move X to A space
    imMoving_Tx1 = vtkfilters.vtiToVts_viaTransform(imMoving, transMatrix=TXA_move2fix)
    # Resample X to Atlas so have same dims
    imMoving_Tx1 = vtkfilters.filterResampleToDataset(imMoving_Tx1, imFixed)
    if DEBUG:
        print(fIO.writeVTKFile(imMoving_Tx1, '/home/fraser/temp/temp2.vti'))
    # Ensure data is in correct format:
    setVTIArrayTo16int(imMoving_Tx1, aName)
    setVTIArrayTo16int(imFixed, aName)
    # Atlas is FIXED
    imMoving_Tx2, Tx2 = deformVTIA_to_VTIB(imFixed, imMoving_Tx1, aName, aName, maxIter=500, subSampleVal=2, blurVal=0.005)
    if DEBUG:
        print(fIO.writeVTKFile(imMoving_Tx2, '/home/fraser/temp/temp3.vti'))
        # surfi = fsitk.transformPolyData(surfAtlas, Tx2)
        # print(fIO.writeVTKFile(surfi, os.path.join(WORKING_DIR, 'AtlasSurfAtDestination.vtp')))
    return imMoving_Tx2, Tx2, TXA_move2fix



def deformPvdToSingleTimePoint(masterF, pvdFile, arrayName,
                               outputDir=None,
                               outputLabel=None,
                               maxIter=500,
                               subSampleVal=None,
                               blurVal=None,
                               surfIn=None,
                               WRITE_TRANSFORM=False,
                               WRITE_IMAGE=False,
                               WATERFALL_REGISTRATION=False,
                               USE_INIT_TX=True,
                               RMSError=0.00001,
                               DEBUG=False):
    """

    :param masterF: file of fixed vti (if int then is ID in pvd)
    :param pvdFile: pvd - vti to register to fixed (fixed should be one of these)
    :param arrayName: e.g. PixelData
    :param outputDir: to write outputs (if None, overwrite)
    :param outputLabel: to write outputs (if None, overwrite)
    :param maxIter: [50]
    :param subSampleVal: Set to subsmaple [None]
    :param blurVal: val in real world coords to blur (FDQ: 0.001 - 0.003)
    :param surfIn: Provide a surafce as polydata - will deform based on deformation field as go
    :param WRITE_TRANSFORM: bool - set true to write out the deformation field as hdf5
    :param WRITE_IMAGE: bool - set true to write out registered images
    :param WATERFALL_REGISTRATION: Will take last registered image and use as fixed for next in line
    :param USE_INIT_TX: Will use last transform field as init transform field [default=True]. Set fasle for regenerate Tx
    :param RMSError: default 1e-5
    :return:
    """
    ## TESTING:
    # vtkToItk = sitkHelper_VTKToITKImage_old
    # itkToVtk = sitkHelper_ITKToVTKImage_old
    vtkToItk = sitkHelper_VTKToITKImage
    itkToVtk = sitkHelper_ITKToVTKImage
    ##
    registeredDict, transformDict, surfDict = {}, {}, {}
    transformDict2 = {}
    if type(pvdFile) == str:
        ffD = fIO.readPVDFileName(pvdFile)
        rr, ff, _ = fIO.pvdGetDataFileRoot_Prefix_and_Ext(pvdFile)
        if outputDir is None:
            outputDir = rr
        if outputLabel is None:
            outputLabel = ff
    else:
        ffD = pvdFile
    timeS = sorted(ffD.keys())
    ffDvals = [ffD[i] for i in timeS]
    try:
        fixedID = int(masterF)
        masterF = ffDvals[fixedID]
    except ValueError:
        fixedID = ffDvals.index(masterF)
    # This allows an initial transform to "chase" the transform steps (WATERFALL_REGISTRATION takes step further)
    idsToRegister = list(range(fixedID, len(timeS))) + list(range(fixedID-1,-1,-1))
    fixed = vtkToItk(masterF, arrayName=arrayName)
    tx = None
    print(idsToRegister)
    for k1, thisID in enumerate(idsToRegister):
        if WATERFALL_REGISTRATION:
            if thisID == (fixedID-1):
                print(thisID, 'Reread %s'%(masterF))
                fixed = vtkToItk(masterF, arrayName=arrayName)
            elif thisID != idsToRegister[0]:
                print(thisID, 'Use img from %d'%(idsToRegister[k1-1]))
                fixed = img
        if (thisID == (fixedID-1))|(not USE_INIT_TX):
            tx = None
        iTime = timeS[thisID]
        iFile = ffD[iTime]
        moving = vtkToItk(iFile, arrayName=arrayName)
        if DEBUG:
            print('Move %s to %s (WATERFALL=%s, USE_INIT_Tx=%s)'%(iFile, masterF, WATERFALL_REGISTRATION, USE_INIT_TX))

        print("---- Loop %d (ID:%d) of %d ----"%(k1, thisID, len(idsToRegister)))
        tx, img = registrationDiffeomorphicDemon(fixed, moving,
                                                 maxIterations=maxIter,
                                                 blurVal=blurVal,
                                                 subsample=subSampleVal,
                                                 initTransform=tx,
                                                 RMSError=RMSError)
        ii = itkToVtk(img, arrayName=arrayName)
        ## RENAME ARRAYS
        vtkfilters.addNpArray(ii, vtkfilters.getScalarsAsNumpy(ii).astype(int), arrayName, SET_SCALAR=True)
        vtkfilters.delArraysExcept(ii, [arrayName])
        if WRITE_IMAGE | DEBUG:
            ff = fIO.writeVTKFile(ii, os.path.join(outputDir, 'Im%d.vti'%(thisID)))
            registeredDict[iTime] = ff
        if WRITE_TRANSFORM:
            tf = os.path.join(outputDir, 'tx%d.hdf5'%(thisID))
            sitk.WriteTransform(tx, tf)
            transformDict[iTime] = tf
            if DEBUG:
                ## This is nice to see the transform vec field in ParaView but does not add anything. hdf5 used to apply transform
                tIm = tx.GetDisplacementField()
                tImVTK = sitkHelper_ITKToVTKImage(tIm, arrayName='TransformField')
                tf2 = fIO.writeVTKFile(tImVTK, os.path.join(outputDir, 'tx%d.vti'%(thisID)))
                transformDict2[iTime] = tf2
        if surfIn is not None:
            if type(surfIn) == str:
                surfVTP = fIO.readVTKFile(surfIn)
            else:
                surfVTP = surfIn
            surfi = transformPolyData(surfVTP, tx)
            surfOut = fIO.writeVTKFile(surfi, os.path.join(outputDir, 'surf%d.vtp'%(thisID)))
            surfDict[iTime] = surfOut
        # del (ii, tx, img)

    if len(registeredDict) > 0:
        print(fIO.writeVTK_PVD_Dict(registeredDict, outputDir, '%s_RegDD'%(outputLabel), 'vti'))
    if len(transformDict) > 0:
        if DEBUG:
            print(fIO.writeVTK_PVD_Dict(transformDict2, outputDir, '%sTransform'%(outputLabel), 'vti'))
        print(fIO.writeVTK_PVD_Dict(transformDict, outputDir, '%s_TransformField'%(outputLabel), 'hdf5'))
    if len(surfDict) > 0:
        print(fIO.writeVTK_PVD_Dict(surfDict, outputDir, '%s'%(outputLabel), 'stl'))


def transformSurfViaTransformField(transFormImagePvd, outputDir, outputLabel, surfIn, extn):
    """
    :param transFormImagePvd:
    :param outputDir:
    :param outputLabel:
    :param surfIn: str or vtp_obj
    :param extn: extn
    :return:
    """
    transDict = fIO.readPVDFileName(transFormImagePvd)
    surfDict = {}
    fOut = None
    if extn.endswith('stl'):
        extn = 'vtp'
    times = sorted(transDict.keys())
    for k1 in range(len(times)):
        tx = sitk.ReadTransform(transDict[times[k1]])
        transField = sitk.DisplacementFieldTransform(tx)
        if type(surfIn) == str:
            surfVTP = fIO.readVTKFile(surfIn)
        else:
            surfVTP = surfIn
        surfi = transformPolyData(surfVTP, transField)
        surfOut = fIO.writeVTKFile(surfi, os.path.join(outputDir, 'surf%d.%s'%(k1, extn)))
        surfDict[times[k1]] = surfOut
    if len(surfDict) > 0:
        fOut = fIO.writeVTK_PVD_Dict(surfDict, outputDir, '%s'%(outputLabel), extn)
        print(fOut)
    return fOut

def transformSurfViaTransformField_Single(transField, surfIn):
    """
    :param transField: is a sitk.DisplacementFieldTransform(tx)
    :param surfIn: str or vtp_obj
    :return: surfOut
    """
    if type(surfIn) == str:
        surfVTP = fIO.readVTKFile(surfIn)
    else:
        surfVTP = surfIn
    surfi = transformPolyData(surfVTP, transField)
    return surfi

def transformViaTransformField_single(iiToTransform, transField, arrayName):
    iiFixed = sitkHelper_VTKToITKImage(iiToTransform)
    iiTransformed = sitk.Resample(iiFixed, iiFixed,
                                  transField,
                                  sitk.sitkLinear,
                                  0.0)
    ii = sitkHelper_ITKToVTKImage(iiTransformed)
    vtkfilters.addNpArray(ii, vtkfilters.getScalarsAsNumpy(ii), arrayName, SET_SCALAR=True)
    vtkfilters.delArraysExcept(ii, [arrayName])
    return ii


def transformViaTransformField(fixedImageFileOrPvd, transFormImagePvd, outputDir, outputLabel, arrayName,
                                            surfIn=None, INVERT_TRANSFORM=False):
    """

    :param fixedImageFileOrPvd:
    :param transFormImagePvd:
    :param outputDir:
    :param outputLabel:
    :param arrayName:
    :param surfIn:
    :param INVERT_TRANSFORM: [False]
    :return:
    """
    try:
        fixedImageDict = fIO.readPVDFileName(fixedImageFileOrPvd)
    except TypeError:
        fixedImageDict = fixedImageFileOrPvd
    timesFixed = sorted(fixedImageDict.keys())
    if len(timesFixed) == 1:
        iiFixed = sitkHelper_VTKToITKImage(fixedImageDict[timesFixed[0]])
    transDict = fIO.readPVDFileName(transFormImagePvd)
    transformdedDict = {} # Keep file names as we write
    surfDict = {}

    times = sorted(transDict.keys())
    for k1 in range(len(times)):
        if len(timesFixed) > 1:
            iTimeFixed = timesFixed[vtkfilters.ftk.getIDOfClosestFloat(times[k1], timesFixed)]
            iiFixed = sitkHelper_VTKToITKImage(fixedImageDict[iTimeFixed], arrayName=arrayName)
        transField = sitk.ReadTransform(transDict[times[k1]])
        if surfIn is not None:
            if type(surfIn) == str:
                surfVTP = fIO.readVTKFile(surfIn)
            else:
                surfVTP = surfIn
            surfi = transformPolyData(surfVTP, transField)
            surfOut = fIO.writeVTKFile(surfi, os.path.join(outputDir, 'surf%d.vtp'%(k1)))
            surfDict[times[k1]] = surfOut
        if INVERT_TRANSFORM:
            transField = invertTransform(sitk.DisplacementFieldTransform(transField))
        iiTransformed = sitk.Resample(iiFixed, iiFixed,
                                       transField,
                                       sitk.sitkLinear,
                                       0.0)
        ii = sitkHelper_ITKToVTKImage(iiTransformed)
        vtkfilters.addNpArray(ii, vtkfilters.getScalarsAsNumpy(ii), arrayName, SET_SCALAR=True)
        vtkfilters.delArraysExcept(ii, [arrayName])
        ff = fIO.writeVTKFile(ii, os.path.join(outputDir, 'Im%d.vti'%(k1)))
        transformdedDict[times[k1]] = ff
    if len(transformdedDict) > 0:
        print(fIO.writeVTK_PVD_Dict(transformdedDict, outputDir, '%s'%(outputLabel), 'vti'))
    if len(surfDict) > 0:
        print(fIO.writeVTK_PVD_Dict(surfDict, outputDir, 'surf%s'%(outputLabel), 'vtp'))


def writeOutTransformField(fIN, outDir, outprefix):
    # This is for testing / debug
    transDict = fIO.readPVDFileName(fIN)
    times = sorted(transDict.keys())
    dOut = {}
    for k1 in range(len(times)):
        tx = sitk.ReadTransform(transDict[times[k1]])
        transField = sitk.DisplacementFieldTransform(tx)
        tIm = transField.GetDisplacementField()
        tImVTK = sitkHelper_ITKToVTKImage(tIm, arrayName='TransformField')
        tf2 = fIO.writeVTKFile(tImVTK, os.path.join(outDir, 'tx%d.vti'%(k1)))
        dOut[times[k1]] = tf2
    fIO.writeVTK_PVD_Dict(dOut, outDir, outprefix, 'vti')

### ====================================================================================================================
### ====================================================================================================================
def biasCorrectionOfImageData_Generalised(imageFile_orImage, arrayName, vtiFileOUT=None, sampleRate=(6,6,6), MedFilter=10, MASK=None):
    """
    This is a generalisied method that does not use the shrink factor approach but rather a
    subsampling and then Med filter
    This was tested and optimised as part of BB pipeline.
    Generalised for use with URO pipeline
    """
    if type(imageFile_orImage) == str:
        ii = fIO.readVTKFile(imageFile_orImage)
    else:
        ii = imageFile_orImage
    dims = ii.GetExtent() # takes care of if already subgrid
    if (type(sampleRate) != list) & (type(sampleRate) != tuple):
        sampleRate = [sampleRate, sampleRate, sampleRate]
    vtkfilters.setArrayAsScalars(ii, arrayName)
    iiSub = vtkfilters.extractVOI(ii, ijkMinMax=dims, sampleRate=sampleRate)
    iiSubMed = vtkfilters.filterVtiMedian(iiSub, MedFilter)
    iiMedSITK = sitkHelper_VTKToITKImage(iiSubMed, arrayName)
    if MASK is None:
        maskImage = sitk.OtsuThreshold(iiMedSITK, 0, 1, 256)
    else:
        MASK = vtkfilters.extractVOI(MASK, ijkMinMax=dims, sampleRate=sampleRate)
        MASK.SetOrigin(iiSubMed.GetOrigin()) # exports from MITK have zero origin
        maskImage = sitkHelper_VTKToITKImage(MASK)
        maskImage = sitk.Cast(maskImage, sitk.sitkUInt8)
    iiMedSITK = sitk.Cast(iiMedSITK, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50] * 4)
    output = corrector.Execute(iiMedSITK)
    # output = corrector.Execute(iiMedSITK, maskImage)
    iiSITK = sitkHelper_VTKToITKImage(ii)
    iiSITK = sitk.Cast(iiSITK, sitk.sitkFloat32)
    log_bias_field = corrector.GetLogBiasFieldAsImage(iiSITK)
    output = iiSITK / sitk.Exp(log_bias_field)
    output = sitk.RescaleIntensity(output)
    output = sitk.Cast(output, sitk.sitkUInt16)
    iiOut = sitkHelper_ITKToVTKImage(output, arrayName)
    if vtiFileOUT is not  None:
        print(fIO.writeVTKFile(iiOut, vtiFileOUT))
    else:
        return iiOut


def biasFieldCorrection(imageFile_orImage, arrayName, numberFittingLevels=4,
                        maxIterations=50, shrinkFactor=None, maskImage=None,
                        arrayNameOut=None):
    if arrayNameOut is None:
        arrayNameOut = arrayName

    # pip install --upgrade --pre SimpleITK --find-links https://github.com/SimpleITK/SimpleITK/releases/tag/latest
    inputImage = sitkHelper_VTKToITKImage(imageFile_orImage, arrayName=arrayName) # , frmt=np.float32)
    image = inputImage
    # maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    if shrinkFactor is not None:
        if not type(shrinkFactor) == list:
            shrinkFactor = [shrinkFactor]
        shrinkFactor = shrinkFactor * inputImage.GetDimension()
        image = sitk.Shrink(inputImage, shrinkFactor)
        if maskImage is not None:
            maskImage = sitk.Shrink(maskImage, shrinkFactor)

    # print(fIO.writeVTKFile(sitkHelper_ITKToVTKImage(maskImage), '/home/fraser/temp/tempM.vti'))
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([maxIterations] * numberFittingLevels)
    if maskImage is not None:
        output = corrector.Execute(image, maskImage)
    else:
        output = corrector.Execute(image)
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    output = inputImage / sitk.Exp( log_bias_field )
    # sitk.WriteImage(output, '/home/fraser/temp/temp.mha') ## DEBUG
    return sitkHelper_ITKToVTKImage(output, arrayName=arrayNameOut)

def normalizeImageFilter(imageFile_orImage, arrayName, arrayNameOut=None):
    if arrayNameOut is None:
        arrayNameOut = arrayName
    inputImage = sitkHelper_VTKToITKImage(imageFile_orImage, arrayName=arrayName, frmt=np.float32)
    nif = sitk.NormalizeImageFilter()
    output = nif.Execute(inputImage)
    return sitkHelper_ITKToVTKImage(output, arrayName=arrayNameOut)


### ====================================================================================================================
###     Bias Correction Tool
### ====================================================================================================================
class BiasCorrectionInteractive(): # THIS IS TOO SLOW 
    def __init__(self):
        self.dicomDir = None
        self.inputImage = None
        self.outputImage = None
        self.fig, self.axs = plt.subplots(2,4)
        self.subsampleLevel = 5
        self.medFilterLevel = 5
        self.slices = None
        self.slicesCorrected = None
        self.plotImage = []
        self.plotImageCorrected = []
        self.plotLinesCorrected = []
        self.maskImage = None # Can set to pass into corrector

    def setInputImage(self, image_or_file):
        if type(image_or_file) == str:
            image_or_file = fIO.readVTKFile(image_or_file)
        self.inputImage = image_or_file
        self.setSlices()
        self.doBiasCorrection()
        self.__setUpPlot()

    def setSlices(self):
        dims = self.inputImage.GetDimensions()
        self.slices = [vtkfilters.extractVOI(self.inputImage, [int(dims[0]/2), int(dims[0]/2)+1, 0, dims[1]-1, 0, dims[2]-1]),
            vtkfilters.extractVOI(self.inputImage, [0, dims[0]-1, int(dims[1]/2), int(dims[1]/2)+1, 0, dims[2]-1]),
            vtkfilters.extractVOI(self.inputImage, [0, dims[0]-1, 0, dims[1]-1, int(dims[2]/2), int(dims[2]/2)+1])]

    def __updatePlot(self):
        print('WORKING (sub=%d, med=%d) ...'%(self.subsampleLevel, self.medFilterLevel))
        self.doBiasCorrection()
        self.__updateImagePlotData()
        self.fig.canvas.draw_idle()
        print('DONE...')

    def __discreteUpdateSliders(self):
        if int(self.subsample_slider.val) != self.subsampleLevel:
            self.subsampleLevel = int(self.subsample_slider.val)
            self.__updatePlot()
        elif int(self.medfilt_slider.val) != self.medFilterLevel:
            self.medFilterLevel = int(self.medfilt_slider.val)
            self.__updatePlot()


    def __setUpPlot(self):
        # # The function to be called anytime a slider's value changes
        def update(val):
            self.__discreteUpdateSliders()

        # Make a horizontal slider to control the subsampel.
        axSubS = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='blue')
        self.subsample_slider = Slider(ax=axSubS, label='Subsample', valmin=1, valmax=10, valinit=5, valstep=1)
        axMef = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor='blue')
        self.medfilt_slider = Slider(ax=axMef, label='MedFilt', valmin=1, valmax=10, valinit=5, orientation="vertical", valstep=1)
        # # register the update function with each slider
        self.subsample_slider.on_changed(update)
        self.medfilt_slider.on_changed(update)
        # INFO
        self.fig.suptitle('Top:ORIG, BOTTOM:BC. R=PA, G=LR, B=SI', fontsize=16)
        ## Add init plot data
        cc = ['r','g','b']
        for k1 in range(3):
            imD = np.mean(vtkfilters.getArrayAsNumpy(self.slices[k1], 'PixelData', RETURN_3D=True), k1)
            imDC = np.mean(vtkfilters.getArrayAsNumpy(self.slicesCorrected[k1], 'PixelData', RETURN_3D=True), k1)
            if k1 == 1:
                imD = imD.T
                imDC = imDC.T
            self.plotImage.append(self.axs[0][k1].imshow(imD))
            self.plotImageCorrected.append(self.axs[1][k1].imshow(imDC, vmin=0, vmax=255))
            print(f'  Plot image {k1}, size: {imD.shape} & {imDC.shape}')
            self.axs[0][k1].set_axis_off()
            self.axs[1][k1].set_axis_off()
            for k2 in range(1,4):
                lD = imD[int(imD.shape[0]*(k2/4)), :]
                lDC = imDC[int(imDC.shape[0]*(k2/4)), :]
                self.axs[0][3].plot(np.linspace(0,1.0, len(lD)), lD, cc[k1])
                self.plotLinesCorrected.append(self.axs[1][3].plot(np.linspace(0,1.0,len(lDC)), lDC, cc[k1]+'--'))
            # self.axs[1][3].set_ylim(self.axs[0][3].get_ylim())
        plt.show()

    def __updateImagePlotData(self):

        for k1 in range(3):
            imD = np.mean(vtkfilters.getArrayAsNumpy(self.slices[k1], 'PixelData', RETURN_3D=True), k1)
            imDC = np.mean(vtkfilters.getArrayAsNumpy(self.slicesCorrected[k1], 'PixelData', RETURN_3D=True), k1)
            self.plotImage[k1].set_array(imD)
            self.plotImageCorrected[k1].set_array(imDC)
            lDC = imDC[int(imDC.shape[0]/2), :]
            self.plotLinesCorrected[k1][0].set_ydata(lDC)
            self.axs[1][3].set_ylim(self.axs[0][3].get_ylim())

    def doBiasCorrection(self):
        subSamepleList = [(1,self.subsampleLevel,self.subsampleLevel), (self.subsampleLevel,1,self.subsampleLevel), (self.subsampleLevel,self.subsampleLevel,1)]
        self.slicesCorrected = [biasCorrectionOfImageData_Generalised(i, arrayName='PixelData', sampleRate=iis, MedFilter=self.medFilterLevel, MASK=self.maskImage) for i, iis in zip(self.slices, subSamepleList)]

    def doFullVolumeCorrection(self, fileOut=None):
        print('DOING FULL CORRECTION')
        iiCorrected = biasCorrectionOfImageData_Generalised(self.inputImage, arrayName='PixelData', sampleRate=self.subsampleLevel, MedFilter=self.medFilterLevel, MASK=self.maskImage)
        if fileOut is not None:
            return fIO.writeVTKFile(iiCorrected, fileOut)
        return iiCorrected


### ====================================================================================================================
### ====================================================================================================================


# ==================================================================
# S T A R T
#
if __name__ == '__main__':
    print('TESTING')
    # iif = '/Volume/KMR_CARDIAC_DATA/KMRC000401/PROCESSED/KMRC000401a-MRA/KMRC000401a-MRA_00005.vti'
    # iiITK = sitkHelper_VTKToITKImage(iif, arrayName='MRA')
    # sitk.WriteImage(iiITK, '/home/fraser/temp/temp1.mha')
    # iiVTK = sitkHelper_ITKToVTKImage(iiITK)
    # print(fIO.writeVTKFile(iiVTK, '/home/fraser/temp/temp1.vti'))
    #
    # bcOut = biasFieldCorrection(iif, maxIterations=100, shrinkFactor=2, arrayName='MRA')
    # print(fIO.writeVTKFile(bcOut, '/home/fraser/temp/temp2.vti'))
