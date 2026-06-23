# fsitk
SITK pipelines and helper functions

This is a repo that has been developed from testing and investigating SimpleITK usage for mdeical image processing. Primarily registration. 

## Installation

Download or clone thos repo and install using pip:

```
cd fsitk
pip install -e .
```

## Examples

IO via fIO from ngawari repo.

### Bias Correction

```python
a3d = np.random.rand(100, 100, 100)
ii = vtkfilters.buildRawImageData([100, 100, 100], [1,1,1], [0,0,0])
vtkfilters.addNpArray(ii, a3d, 'PixelData', SET_SCALAR=True)
iiBC = fsitk.biasCorrectionOfImageData_Generalised(ii, 'PixelData', sampleRate=5, MedFilter=7)
```

### Diffeomorphic Demons Registration

```python
vtiDict = {}
for i in range(10):
    a3d = np.random.rand(100, 100, 100)
    ii = vtkfilters.buildRawImageData([100, 100, 100], [1,1,1], [0,0,0])
    vtkfilters.addNpArray(ii, a3d, 'PixelData', SET_SCALAR=True)
    vtiDict[i] = ii

fsitk.deformPvdToSingleTimePoint(0, 
                                vtiDict, 
                                'PixelData', 
                                outputDir='output',
                                outputLabel="Test_DReg",
                                maxIter=50,
                                WRITE_TRANSFORM=True, WRITE_IMAGE=False,
                                subSampleVal=2, blurVal=0.005, DEBUG=False)
# Increase subSample to speed up at lower accuracy. 
# Reduce blurVal for improved accuracy - but may introduce more noise. 
```

### Deform Surfaces
```python 
fixedSurf = fIO.readVTKFile('fixed.vtp')
fOUT = fsitk.transformSurfViaTransformField('output/Test_DReg.pvd',
                                outputDir='output',
                                outputLabel='Test_DReg_Surf',
                                surfIn='fixed.vtp', 
                                extn="stl")
print(fOUT) # PVD file with fixed.vtp deformed by transformation field. 
```