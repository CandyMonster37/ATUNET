import os
import numpy as np
import pandas as pd
from imageio.v2 import imread, imwrite


def image_read(fname, data_type='Radar'):
    """
    :param fname: path of the image to be read
    :param data_type: 'Radar' or 'Wind' or 'Precip'
    :return: ndarray of the image
    """
    # 雷达回波 Radar factor = 70
    # 风速 Wind factor = 35
    # 降水 Precip factor = 10
    dic = {'Radar': 70, 'Wind': 35, 'Precip': 10}
    factor = dic[data_type]
    image = np.array(imread(fname) / 255 * factor).astype('float32')
    return image


def image_write(image, write_path='./Radar/', data_type='Radar'):
    # 雷达回波 Radar factor = 70
    # 风速 Wind factor = 35
    # 降水 Precip factor = 10
    dic = {'Radar': 70, 'Wind': 35, 'Precip': 10}
    factor = dic[data_type]
    image = np.clip(np.array(image), 0, factor) / factor * 255.  # 放缩回原0-255区间
    imwrite(write_path, image)


def proc_img_name(target_dir='../data/Train', mode='Radar', idx_file='../data/Train.csv', test=False):
    input_imgs = []
    output_imgs = [] if not test else None
    idx_loader = pd.read_csv(idx_file, header=None)
    inputs_idx = idx_loader.iloc[:, :20]
    inputs_idx = np.array(inputs_idx).tolist()
    for seq in inputs_idx:
        cont_seq = []
        for image_idx in seq:
            img_name = os.path.join(target_dir, mode, mode.lower() + '_' + image_idx)
            cont_seq.append(img_name)
        input_imgs.append(cont_seq)
    if not test:
        outputs_idx = idx_loader.iloc[:, 20:]
        outputs_idx = np.array(outputs_idx).tolist()
        for seq in outputs_idx:
            cont_seq = []
            for image_idx in seq:
                img_name = os.path.join(target_dir, mode, mode.lower() + '_' + image_idx)
                cont_seq.append(img_name)
            output_imgs.append(cont_seq)

    return input_imgs, output_imgs


def batch_generator(input_img_paths: list,
                    output_img_paths=None,
                    data_type='Radar',
                    batch_size=128,
                    test_set=False):
    """
    :param input_img_paths: [[path, path, ...], ...], (len, 20)
    :param output_img_paths: None if use test data; else [[path, path, ...], ...]. (len, 20)
    :param data_type: 'Radar' or 'Wind' or 'Precip'
    :param batch_size: 2 ^ n
    :param test_set: True if use test data; else False
    :return:
    """
    inputs_batch = []
    outputs_batch = []
    data_size = len(input_img_paths)
    i = 0
    while True:
        one_seq_in, one_seq_out = [], []
        for image in input_img_paths[i]:
            img_data = image_read(image, data_type=data_type).tolist()
            one_seq_in.append(img_data)
        inputs_batch.append(one_seq_in)
        if not test_set:
            # 只要不是测试集数据
            for image in output_img_paths[i]:
                img_data = image_read(image, data_type=data_type).tolist()
                one_seq_out.append(img_data)
            outputs_batch.append(one_seq_out)
        now_size = len(inputs_batch)
        i = (i + 1) % data_size
        if now_size >= batch_size:
            inputs = np.asarray(inputs_batch).astype('float32')
            outputs = np.asarray(outputs_batch).astype('float32')
            inputs_batch, outputs_batch = [], []
            yield inputs, outputs


if __name__ == '__main__':
    test = '../data/TestA/Radar/radar_31218.png'
    img = image_read(test, data_type='Radar')
    print(img)
    print(img.shape)
    dirs = '../data/Train'
    mod = 'Radar'
    id_file = '../data/Train.csv'
    input_imgnames, output_imgnames = proc_img_name(target_dir=dirs, mode=mod, idx_file=id_file, test=False)
    my_generator = batch_generator(input_imgnames, output_imgnames,
                                   data_type='Radar', batch_size=32, test_set=False)
    print(next(my_generator)[0].shape)
