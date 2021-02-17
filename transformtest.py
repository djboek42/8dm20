import imageio
import elastix
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scrollview import ScrollView
from transform_parameter_editor import TransformParameterFileEditor
from metrics import dice_coef, sensitivity, specificity

## Als je de bspline al toegepast hebt en met de resultaten nog aparte dingen wilt proberen met transformix

ELASTIX_PATH = os.path.join(r'C:\Users\s145680\AppData\Local\Programs\Python\Python39\8MD20\elastix.exe')
TRANSFORMIX_PATH = os.path.join(r'C:\Users\s145680\AppData\Local\Programs\Python\Python39\8MD20\transformix.exe')

if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

# Make a results directory if non exists
if os.path.exists('results') is False:
    os.mkdir('results')

unseen = 'p119'
patient = 'p120'

#unseen mri
image_path_unseen = os.path.join(unseen, 'mr_bffe.mhd')
itk_image_unseen = sitk.ReadImage(image_path_unseen)
image_array_unseen = sitk.GetArrayFromImage(itk_image_unseen)

#opr: original prostate
image_path_opr = os.path.join(unseen, 'prostaat.mhd')
itk_image_opr = sitk.ReadImage(image_path_opr)
image_array_opr = sitk.GetArrayFromImage(itk_image_opr)

pr_image_path = os.path.join(patient, 'prostaat.mhd')
itk_image_pr = sitk.ReadImage(pr_image_path)
image_array_pr = sitk.GetArrayFromImage(itk_image_pr)

mr_image_path = os.path.join(patient, 'mr_bffe.mhd')
itk_image_mr = sitk.ReadImage(mr_image_path)
image_array_mr = sitk.GetArrayFromImage(itk_image_mr)

final_result_path = os.path.join('results', 'result.1.mhd')
transform_path0 = os.path.join('results', 'TransformParameters.0.txt')
transform_path1 = os.path.join('results', 'TransformParameters.1.txt')
final_transform_path = os.path.join('results', 'transform_pathfinal.txt')

TransformParameterFileEditor(transform_path1, transform_path0, final_transform_path).modify_transform_parameter_file()

tr = elastix.TransformixInterface(parameters=final_transform_path,
                                  transformix_path=TRANSFORMIX_PATH, )

fig, (ax8, ax9, ax10) = plt.subplots(1, 3)

ScrollView(image_array_opr).plot(ax8, vmin=0, vmax=1)

transformed_pr_path = tr.transform_image(pr_image_path, output_dir = r'results')
itk_image_tpr = sitk.ReadImage(transformed_pr_path)
image_array_tpr = sitk.GetArrayFromImage(itk_image_tpr)

ScrollView(image_array_tpr).plot(ax9, vmin=0, vmax=1)

otpr = abs(image_array_tpr-image_array_opr)
ScrollView(otpr).plot(ax10, vmin=0, vmax=1)

DSC = dice_coef(image_array_opr, image_array_tpr)
SNS = sensitivity(image_array_opr, image_array_tpr)
SPC = specificity(image_array_opr, image_array_tpr)

print("Dice coefficient is %.2f, sensitivity is %.2f, specificity is %.2f" % (DSC, SNS, SPC))

plt.show()
