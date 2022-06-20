import os
import tarfile


def un_tar(file_name, output_root='dataset/train'):
    # untar zip file to folder whose name is same as tar file
    tar = tarfile.open(file_name)
    names = tar.getnames()

    file_name = os.path.basename(file_name)
    extract_dir = os.path.join(output_root, file_name.split('.')[0])

    # create folder if nessessary
    if os.path.isdir(extract_dir):
        pass
    else:
        os.mkdir(extract_dir)

    for name in names:
        tar.extract(name, extract_dir)
    tar.close()


def untar_traintar(traintar='./traintar'):
    """
    untar images from traintar and save in corresponding folders
    organize like:
    /train
       /n01440764
           images
       /n01443537
           images
        .....
    """
    files = os.listdir(traintar)
    for file in files:
        un_tar(os.path.join(traintar, file))
untar_traintar('./dataset/train/ILSVRC2012_img_train')