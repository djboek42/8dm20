import imageio
import elastix
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scrollview import ScrollView
from transform_parameter_editor import TransformParameterFileEditor
from metrics import dice_coef, sensitivity, specificity
import statistics
from normalization import normalization

ELASTIX_PATH = os.path.join(r'C:\Users\s145680\AppData\Local\Programs\Python\Python39\8MD20\elastix.exe')
TRANSFORMIX_PATH = os.path.join(r'C:\Users\s145680\AppData\Local\Programs\Python\Python39\8MD20\transformix.exe')

if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

unseen = 'p127'
patients = ['p102', 'p107', 'p108', 'p109', 'p115', 'p116', 'p117', 'p119', 'p120', 'p125', 'p127', 'p128', 'p129', 'p133', 'p135']
unseen_index = patients.index(unseen)

unseen_folder = 'results{}'.format(unseen)
# Make a results directory if non exists
if not os.path.exists(unseen_folder):
    os.mkdir(unseen_folder)

# Get unseen mri and segmentations
unseen_image_path = os.path.join(unseen, 'mr_bffe.mhd')
image_array_unseen = sitk.GetArrayFromImage(sitk.ReadImage(unseen_image_path))

opr_image_path = os.path.join(unseen, 'prostaat.mhd')
image_array_opr = sitk.GetArrayFromImage(sitk.ReadImage(opr_image_path))

# Define a new elastix object 'el' with the correct path to elastix
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

DSC = []
SNS = []
SPC = []
finalMI = []

# Execute the registrations.
for i, patient in enumerate(patients):

    # Reading patient segmentation and mri
    pr_image_path = os.path.join(patient, 'prostaat.mhd')
    mr_image_path = os.path.join(patient, 'mr_bffe.mhd')

    # Everything for this registration will be put in result_path
    pat_folder = 'result_unseen{}_{}'.format(unseen, patient)
    result_path = os.path.join(unseen_folder, pat_folder)

    # Create folder for this one registration
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    # Perform registration with affine and bspline
    el.register(
       fixed_image=unseen_image_path,
       moving_image=mr_image_path,
       parameters=[os.path.join('parameterfiles', 'parameters_affine.txt'), os.path.join('parameterfiles', 'parameters_bspline.txt')],
       output_dir=result_path)

    final_result_path = os.path.join(result_path, 'result.1.mhd')
    transform_path = os.path.join(result_path, 'TransformParameters.1.txt')

    transform_path0 = os.path.join(result_path, 'TransformParameters.0.txt')
    transform_path1 = os.path.join(result_path, 'TransformParameters.1.txt')
    final_transform_path = os.path.join(result_path, 'transform_pathfinal.txt')

    # Change FinalBSplineInterpolationOrder to 0 for binary mask transformation
    TransformParameterFileEditor(transform_path1, transform_path0, final_transform_path).modify_transform_parameter_file()

    # Make a new transformix object tr with the CORRECT PATH to transformix
    tr = elastix.TransformixInterface(parameters=final_transform_path,
                                      transformix_path=TRANSFORMIX_PATH)

    transformed_pr_path = tr.transform_image(pr_image_path, output_dir=result_path)
    image_array_tpr = sitk.GetArrayFromImage(sitk.ReadImage(transformed_pr_path))

    log_path = os.path.join(result_path, 'IterationInfo.1.R3.txt')
    log = elastix.logfile(log_path)

    DSC.append(dice_coef(image_array_opr, image_array_tpr))
    SNS.append(sensitivity(image_array_opr, image_array_tpr))
    SPC.append(specificity(image_array_opr, image_array_tpr))
    finalMI.append(statistics.mean(log['metric'][-50:-1]))

fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.scatter(finalMI,DSC)
ax1.set_title("DSC")
ax2.scatter(finalMI,SNS)
ax2.set_title("SNS")
ax3.scatter(finalMI,SPC)
ax3.set_title("SPC")
plt.show()
