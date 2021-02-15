import imageio
import elastix
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scrollview import ScrollView

ELASTIX_PATH = os.path.join(r'C:\Users\s145680\AppData\Local\Programs\Python\Python39\8MD20\elastix.exe')
TRANSFORMIX_PATH = os.path.join(r'C:\Users\s145680\AppData\Local\Programs\Python\Python39\8MD20\transformix.exe')

if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')

# Make a results directory if non exists
if os.path.exists('linear_results') is False:
    os.mkdir('linear_results')

if os.path.exists('nonlinear_results') is False:
    os.mkdir('nonlinear_results')

unseen = 'p102'
patient = 'p135'

unseen_image_path = os.path.join(unseen, 'mr_bffe.mhd')
itk_image_unseen = sitk.ReadImage(unseen_image_path)
image_array_unseen = sitk.GetArrayFromImage(itk_image_unseen)

opr_image_path = os.path.join(unseen, 'prostaat.mhd')
itk_image_opr = sitk.ReadImage(opr_image_path)
image_array_opr = sitk.GetArrayFromImage(itk_image_opr)

pr_image_path = os.path.join(patient, 'prostaat.mhd')
itk_image_pr = sitk.ReadImage(pr_image_path)
image_array_pr = sitk.GetArrayFromImage(itk_image_pr)

mr_image_path = os.path.join(patient, 'mr_bffe.mhd')
itk_image_mr = sitk.ReadImage(mr_image_path)
image_array_mr = sitk.GetArrayFromImage(itk_image_mr)

overlay = image_array_mr-7000*image_array_pr

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(25, 5))

ScrollView(image_array_unseen).plot(ax1, cmap='gray')
ScrollView(overlay).plot(ax2, cmap='gray')
ax1.set_title('Unseen')
ax2.set_title('Prostate overlay')

# Define a new elastix object 'el' with the correct path to elastix
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)
#
# Execute the registration. Make sure the paths below are correct, and
# that the results folder exists from where you are running this script
el.register(
   fixed_image=unseen_image_path,
   moving_image=mr_image_path,
   parameters=[os.path.join('parameterfiles', 'parameters_affine.txt')],
   output_dir=r'linear_results')

intermediate_result_path = os.path.join('linear_results', 'result.0.mhd')

el.register(
    fixed_image=unseen_image_path,
    moving_image=intermediate_result_path,
    parameters=[os.path.join('parameterfiles', 'parameters_bspline.txt')],
    output_dir=r'nonlinear_results')

final_result_path = os.path.join('nonlinear_results', 'result.0.mhd')
transform_path = os.path.join('nonlinear_results', 'TransformParameters.0.txt')
#

itk_image_result_aff = sitk.ReadImage(intermediate_result_path)
image_array_transformed_aff = sitk.GetArrayFromImage(itk_image_result_aff)
ScrollView(image_array_transformed_aff).plot(ax3, cmap='gray')
ax3.set_title('Affine transformed')

for i in range(4):
    log_path = os.path.join('nonlinear_results', 'IterationInfo.0.R{}.txt'.format(i))
    log = elastix.logfile(log_path)
    # Plot the 'metric' against the iteration number 'itnr'
    ax5.plot(log['itnr'], log['metric'])
plt.legend(['Resolution {}'.format(i) for i in range(4)])

itk_image_result_bsp = sitk.ReadImage(final_result_path)
image_array_transformed_bsp = sitk.GetArrayFromImage(itk_image_result_bsp)
ScrollView(image_array_transformed_bsp).plot(ax4, cmap='gray')
ax4.set_title('Affine&Bspline\n transformed')


# Make a new transformix object tr with the CORRECT PATH to transformix
tr = elastix.TransformixInterface(parameters=transform_path,
                                  transformix_path=TRANSFORMIX_PATH)

# Transform a new image with the transformation parameters
transformed_image_path = tr.transform_image(mr_image_path, output_dir=r'nonlinear_results')

jacobian_matrix_path = tr.jacobian_matrix(output_dir=r'nonlinear_results')
jacobian_determinant_path = tr.jacobian_determinant(output_dir=r'nonlinear_results')

spatialJ_path = os.path.join('nonlinear_results', 'spatialJacobian.mhd')
itk_image_J = sitk.ReadImage(spatialJ_path)
image_array_J = sitk.GetArrayFromImage(itk_image_J)
J_binarized = image_array_J > 0

fig, (ax6, ax7) = plt.subplots(1,2)
ScrollView(J_binarized).plot(ax6, vmin=0, vmax=1)
plt.title('Jacobian\ndeterminant')

ScrollView(image_array_J).plot(ax7)
ax7.set_title('Jacobian\ndeterminant')


fig, (ax8, ax9, ax10) = plt.subplots(1, 3)
ScrollView(image_array_opr).plot(ax8, vmin=0, vmax=1)

transformed_pr_path = tr.transform_image(pr_image_path, output_dir=r'nonlinear_results')
itk_image_tpr = sitk.ReadImage(transformed_pr_path)
image_array_tpr = sitk.GetArrayFromImage(itk_image_tpr)

ScrollView(image_array_tpr).plot(ax9, vmin=0, vmax=1)

otpr = abs(image_array_tpr-image_array_opr)
ScrollView(otpr).plot(ax10, vmin=0, vmax=1)

plt.show()
