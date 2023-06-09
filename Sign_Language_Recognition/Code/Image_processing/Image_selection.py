import os
import numpy as np
import shutil
import random

root_dir = 'images/' 
classes_dir = ['01_palm/', '02_l/', '03_up/', '04_fist_moved/', '05_down/', '06_index/', '07_ok/', '08_palm_m/', '09_c/', '10_palm_u/', '11_heavy/', '12_hang/', '13_two/', '14_three/', '15_four/', '16_five/']

val_ratio = 0.1
test_ratio = 0.15

for cls in classes_dir:
    os.makedirs(root_dir +'train/' + cls)
    os.makedirs(root_dir +'val/' + cls)
    os.makedirs(root_dir +'test/' + cls)


    # Creating partitions of the data after shuffeling
    src = root_dir+'processed/' + cls # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                            [int(len(allFileNames)* (1 - (val_ratio + test_ratio))), 
                                                            int(len(allFileNames)* (1 - test_ratio))])


    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir +'train/' + cls)

    for name in val_FileNames:
        shutil.copy(name, root_dir +'val/' + cls)

    for name in test_FileNames:
        shutil.copy(name, root_dir +'test/' + cls)