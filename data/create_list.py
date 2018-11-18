"""Creat list file of images and labels"""

import os

def create_list_core(srcPath, outputFile):
    """create list file of all the files in srcPath """
    files = os.listdir(srcPath)
    files.sort()
    with open(outputFile, 'w') as f:
        for file in files:
            file_path = os.path.join(srcPath, file)
            f.write(file_path + '\n')


def create_list_icdar2015ch4():
    dataset_name = 'icdar2015ch4'
    dirs = os.listdir(dataset_name)
    assert (len(dirs) == 4), 'data directories should be 4'
    dirs.sort()
    for dir in dirs:
        srcPath = os.path.join(dataset_name, dir)
        outputFile = os.path.join(dataset_name, dir + '.txt')
        print('Processing {}'.format(srcPath))
        create_list_core(srcPath, outputFile)
        print('Finish processing {}\n'.format(srcPath))


if __name__ == '__main__':
    create_list_icdar2015ch4()

