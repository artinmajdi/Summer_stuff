import multiresolutionimageinterface as mir
import matplotlib.pyplot as plt
reader = mir.MultiResolutionImageReader()


def readImage(nodeInd,patientDir,level,path_size):

    Dir = '/media/groot/Seagate Backup Plus Drive/dataset/camelyon17'

    reader = mir.MultiResolutionImageReader()
    name = patientDir + '_node_' + str(nodeInd) +'.tif'
    mr_image = reader.open(Dir + '/' + patientDir + '/' + name)

    ds = mr_image.getLevelDownsample(level)
    image_patch = mr_image.getUCharPatch(int(568 * ds), int(732 * ds), path_size[1], path_size[0], level)

    return image_patch

def AnnoationRead(nodeInd,patientDir,level):

    Dir = '/media/groot/Seagate Backup Plus Drive/dataset'
    name = patientDir + '_node_' + str(nodeInd)

    mr_image = reader.open(Dir + '/' + patientDir + '/' + name +'.tif')
    annotation_list = mir.AnnotationList()
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(Dir + '/' + patientDir + '/lesion_annotations/' + name + '.xml')
    xml_repository.load()
    annotation_mask = mir.AnnotationToMask()

    camelyon17_type_mask = True
    label_map = {'metastases': 1, 'normal': 2} if camelyon17_type_mask else {'_0': 1, '_1': 1, '_2': 0}
    conversion_order = ['metastases', 'normal'] if camelyon17_type_mask else  ['_0', '_1', '_2']
    output_path = Dir + '/' + patientDir + '/lesion_annotations/'
    annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(), mr_image.getSpacing(), label_map, conversion_order)

    return annotation_mask

path_size = [4000,5000]
nodeInd = 0 # 0:4
level = 2
patientDir = 'patient_000'
im = readImage(nodeInd,patientDir,level,path_size)
# annotation_mask = AnnoationRead(nodeInd,patientDir,level)


print(im.shape)
plt.imshow(im)
plt.show()


print(im.max())
print(im.min())
