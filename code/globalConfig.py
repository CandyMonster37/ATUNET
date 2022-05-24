import os

batch_size = 4
seq_len = 20
head_nums = 4
epochs = 1000
out_base_dir = './output'

train_dir = '../data/Train'
valid_dir = '../data/TestA'
test_dir = '../data/TestB/TestB1'
train_ind = '../data/Train.csv'
valid_ind = '../data/TestA.csv'

mode_radar = 'Radar'
mode_wind = 'Wind'
mode_precip = 'Precip'

if not os.path.exists(out_base_dir):
    os.mkdir(out_base_dir)

if __name__ == '__main__':
    pass
