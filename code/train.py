import globalConfig
import os
from model import ATUNET, dice_coef_loss
from utils import proc_img_name, batch_generator
from warmup import AdamWarmup, calc_train_steps
import tensorflow as tf
import json

# 定义公共参数
seq = globalConfig.seq
batch_size = globalConfig.batch_size
head_nums = globalConfig.head_nums
epochs = globalConfig.epochs
save_summary = False
use_trained = False
# 模型种类
radar = globalConfig.mode_radar
wind = globalConfig.mode_wind
precip = globalConfig.mode_precip

mode = precip
# 模型保存位置
save_dir = os.path.join(globalConfig.out_base_dir, 'ckpt_' + mode)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'model_' + mode + '.h5')
# 数据集位置
train_dir = globalConfig.train_dir
train_ind_file = globalConfig.train_ind
valid_dir = globalConfig.valid_dir
valid_ind_file = globalConfig.valid_ind

train_in_names, train_out_names = proc_img_name(target_dir=train_dir, mode=mode,
                                                idx_file=train_ind_file, istest=False)
train_generator = batch_generator(train_in_names, train_out_names,
                                  data_type=mode, batch_size=batch_size, istest_set=False)
train_size = len(train_in_names)
batches_per_epoch = train_size // batch_size + 1

valid_in_names, valid_out_names = proc_img_name(target_dir=valid_dir, mode=mode,
                                                idx_file=valid_ind_file, istest=False)
valid_generator = batch_generator(valid_in_names, valid_out_names,
                                  data_type=mode, batch_size=batch_size, istest_set=False)

if use_trained and os.path.exists(save_path):
    model = tf.keras.models.load_model(save_path)
else:
    model = ATUNET(head_nums=head_nums, seq=seq)
    model.save(save_path)
total_steps, warmup_steps = calc_train_steps(
    num_example=train_size,
    batch_size=batch_size,
    epochs=800,
    warmup_proportion=0.1,
)
model.compile(
    optimizer=AdamWarmup(
        decay_steps=total_steps,
        warmup_steps=warmup_steps,
        learning_rate=1e-3,
        weight_decay=0,
        weight_decay_pattern=None,
    ),
    loss=dice_coef_loss
)

if save_summary:
    import sys
    inputs_x = tf.random.normal((batch_size, 20, 480, 560, 1))
    model(tf.keras.Input(shape=inputs_x.shape[1:], batch_size=batch_size))
    model.summary()
    del inputs_x
    orig_stdout = sys.stdout
    f = open(os.path.join(save_dir, 'model_' + mode + '_summary.txt'), 'w')
    sys.stdout = f
    print(model.summary())
    sys.stdout = orig_stdout
    f.close()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=save_dir + '/train_logs')
bkpath = os.path.join(save_dir, 'model_' + mode)
check_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=bkpath + '_{epoch}.h5',
    save_best_only=True,
    monitor='val_loss',
    verbose=2
)
history = model.fit_generator(train_generator, steps_per_epoch=batches_per_epoch, epochs=epochs,
                              callbacks=[early_stop, tensorboard], verbose=2, validation_data=valid_generator,
                              validation_freq=2, validation_steps=5)
model.save(save_path)
with open(os.path.join(save_dir, 'train_history.json'), 'w') as f:
    json.dump(history.history, f)



if __name__ == '__main__':
    pass
