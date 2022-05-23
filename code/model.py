import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class double_conv2d_bn(keras.Model):
    def __init__(self, out_channels, kernel_size=3, strides=(1, 1), padding='valid'):
        super(double_conv2d_bn, self).__init__()
        self.conv1 = layers.Conv2D(filters=out_channels,
                                   kernel_size=kernel_size,
                                   strides=strides, padding=padding, use_bias=False)
        self.conv2 = layers.Conv2D(filters=out_channels,
                                   kernel_size=kernel_size,
                                   strides=strides, padding=padding, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    def call(self, x):
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = tf.nn.relu(self.bn2(self.conv2(out)))
        return out


class deconv2d_bn(keras.Model):
    def __init__(self, out_channels, kernel_size=3, strides=(1, 1), output_padding=None):
        super(deconv2d_bn, self).__init__()
        self.conv1 = layers.Conv2DTranspose(filters=out_channels,
                                            kernel_size=kernel_size,
                                            output_padding=output_padding,
                                            strides=strides, use_bias=False)
        self.bn1 = layers.BatchNormalization()

    def call(self, x):
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        return out


class Net(keras.Model):
    def __init__(self, head_nums=6, seq=20):
        super(Net, self).__init__()
        self.seq = seq

        # downscale for 4 times
        self.layer1_conv = double_conv2d_bn(out_channels=32)
        self.layer2_conv = double_conv2d_bn(out_channels=64)
        self.layer3_conv = double_conv2d_bn(out_channels=128)
        self.layer4_conv = double_conv2d_bn(out_channels=256)
        self.layer5_conv = double_conv2d_bn(out_channels=512)

        # Multi-Head Self-Attention for 5 times
        self.atten1 = layers.MultiHeadAttention(num_heads=head_nums, key_dim=3, attention_axes=(2, 3, 4))
        self.atten2 = layers.MultiHeadAttention(num_heads=head_nums, key_dim=3, attention_axes=(2, 3, 4))
        self.atten3 = layers.MultiHeadAttention(num_heads=head_nums, key_dim=3, attention_axes=(2, 3, 4))
        self.atten4 = layers.MultiHeadAttention(num_heads=head_nums, key_dim=3, attention_axes=(2, 3, 4))
        self.atten5 = layers.MultiHeadAttention(num_heads=head_nums, key_dim=3, attention_axes=(2, 3, 4))

        # upscale for 4 times
        self.deconv1_1 = deconv2d_bn(out_channels=256, strides=(1, 1))
        self.deconv1_2 = deconv2d_bn(out_channels=256, strides=(1, 1))
        self.deconv1_3 = deconv2d_bn(out_channels=256, strides=(2, 2))

        self.deconv2_1 = deconv2d_bn(out_channels=128, strides=(1, 1))
        self.deconv2_2 = deconv2d_bn(out_channels=128, strides=(1, 1))
        self.deconv2_3 = deconv2d_bn(out_channels=128, strides=(2, 2))

        self.deconv3_1 = deconv2d_bn(out_channels=64, strides=(1, 1))
        self.deconv3_2 = deconv2d_bn(out_channels=64, strides=(1, 1))
        self.deconv3_3 = deconv2d_bn(out_channels=64, strides=(2, 2))

        self.deconv4_1 = deconv2d_bn(out_channels=32, strides=(1, 1))
        self.deconv4_2 = deconv2d_bn(out_channels=32, strides=(1, 1))
        self.deconv4_3 = deconv2d_bn(out_channels=32, strides=(2, 2))

        self.deconv5_1 = deconv2d_bn(out_channels=16, strides=(1, 1))
        self.deconv5_2 = deconv2d_bn(out_channels=8, strides=(1, 1))

        # conv after upscale
        self.deconv1_conv1 = layers.Conv2D(filters=256, kernel_size=2)
        self.deconv1_conv2 = layers.Conv2D(filters=256, kernel_size=1)

        self.deconv2_conv = layers.Conv2D(filters=128, kernel_size=1)

        self.deconv3_conv1 = layers.Conv2D(filters=64, kernel_size=2)
        self.deconv3_conv2 = layers.Conv2D(filters=64, kernel_size=1)

        self.deconv4_conv1 = layers.Conv2D(filters=32, kernel_size=2)
        self.deconv4_conv2 = layers.Conv2D(filters=32, kernel_size=1)

        self.deconv5_conv = layers.Conv2D(filters=1, kernel_size=1)

        self.sigmoid = tf.nn.sigmoid

    def call(self, inputs):
        # inputs: (b, 20, 480, 560, 1)

        # downscale 1
        conv1 = self.layer1_conv.call(inputs)
        # conv1: (b, 20, 476, 556, 32), saved as residual
        conv1_reshape = tf.reshape(conv1, (-1, conv1.shape[-3], conv1.shape[-2], conv1.shape[-1]))
        pool1 = layers.AveragePooling2D(pool_size=(2, 2), input_shape=conv1_reshape.shape[2:-1])
        conv_pool_1 = pool1(conv1_reshape)
        del conv1_reshape
        conv_pool_1 = tf.reshape(conv_pool_1,
                                 (-1, self.seq, conv_pool_1.shape[-3], conv_pool_1.shape[-2], conv_pool_1.shape[-1]))
        # conv_pool_1: (b, 20, 238, 278, 32)

        # downscale 2
        conv2 = self.layer2_conv.call(conv_pool_1)
        # conv2: (b, 20, 234, 274, 64), saved as residual
        del conv_pool_1
        conv2_reshape = tf.reshape(conv2, (-1, conv2.shape[-3], conv2.shape[-2], conv2.shape[-1]))
        pool2 = layers.AveragePooling2D(pool_size=(2, 2), input_shape=conv2_reshape.shape[2:-1])
        conv_pool_2 = pool2(conv2_reshape)
        del conv2_reshape
        conv_pool_2 = tf.reshape(conv_pool_2,
                                 (-1, self.seq, conv_pool_2.shape[-3], conv_pool_2.shape[-2], conv_pool_2.shape[-1]))
        # conv_pool_2: (b, 20, 117, 137, 64)

        # downscale 3
        conv3 = self.layer3_conv.call(conv_pool_2)
        # conv3: (1, 20, 113, 133, 128), saved as residual
        del conv_pool_2
        conv3_reshape = tf.reshape(conv3, (-1, conv3.shape[-3], conv3.shape[-2], conv3.shape[-1]))
        pool3 = layers.AveragePooling2D(pool_size=(2, 2), input_shape=conv3_reshape.shape[2:-1])
        conv_pool_3 = pool3(conv3_reshape)
        del conv3_reshape
        conv_pool_3 = tf.reshape(conv_pool_3,
                                 (-1, self.seq, conv_pool_3.shape[-3], conv_pool_3.shape[-2], conv_pool_3.shape[-1]))
        # conv_pool_3: (1, 20, 56, 66, 128)

        # downscale 4
        conv4 = self.layer4_conv.call(conv_pool_3)
        # conv4: (1, 20, 52, 62, 256), saved as residual
        del conv_pool_3
        conv4_reshape = tf.reshape(conv4, (-1, conv4.shape[-3], conv4.shape[-2], conv4.shape[-1]))
        pool4 = layers.AveragePooling2D(pool_size=(2, 2), input_shape=conv4_reshape.shape[2:-1])
        conv_pool_4 = pool4(conv4_reshape)
        del conv4_reshape
        conv_pool_4 = tf.reshape(conv_pool_4,
                                 (-1, self.seq, conv_pool_4.shape[-3], conv_pool_4.shape[-2], conv_pool_4.shape[-1]))
        # conv_pool_4: (1, 20, 26, 31, 256)

        # conv and attention at bottom
        conv5 = self.layer5_conv.call(conv_pool_4)
        # conv5: (1, 20, 22, 27, 512)
        del conv_pool_4
        conv5 = self.atten5(conv5, conv5)
        # conv5: (1, 20, 22, 27, 512)

        # upscale, attention 4
        conv5 = tf.reshape(conv5, (-1, conv5.shape[-3], conv5.shape[-2], conv5.shape[-1]))
        conv5 = self.deconv1_1.call(conv5)  # (20, 24, 29, 256)
        conv5 = self.deconv1_2.call(conv5)  # (20, 26, 31, 256)
        conv5 = self.deconv1_3.call(conv5)  # (20, 53, 63, 256)
        conv5 = tf.reshape(conv5, (-1, self.seq, conv5.shape[-3], conv5.shape[-2], conv5.shape[-1]))
        conv5 = self.deconv1_conv1(conv5)  # (1, 20, 52, 62, 256)
        conv5 = tf.concat([conv5, conv4], axis=-1)
        del conv4
        conv5 = self.deconv1_conv2(conv5)
        up_atten4 = self.atten4(conv5, conv5)  # (1, 20, 52, 62, 256)
        del conv5

        # upscale, attention 3
        up_atten4 = tf.reshape(up_atten4, (-1, up_atten4.shape[-3], up_atten4.shape[-2], up_atten4.shape[-1]))
        up_atten4 = self.deconv2_1.call(up_atten4)  # (20, 54, 64, 128)
        up_atten4 = self.deconv2_2.call(up_atten4)  # (20, 56, 66, 128)
        up_atten4 = self.deconv2_3.call(up_atten4)  # (20, 113, 133, 128)
        up_atten4 = tf.reshape(up_atten4, (-1, self.seq, up_atten4.shape[-3], up_atten4.shape[-2], up_atten4.shape[-1]))
        up_atten4 = tf.concat([up_atten4, conv3], axis=-1)
        del conv3
        up_atten3 = self.atten3(up_atten4, up_atten4)  # (1, 20, 113, 133, 128)
        del up_atten4

        # upscale, attention 2
        up_atten3 = tf.reshape(up_atten3, (-1, up_atten3.shape[-3], up_atten3.shape[-2], up_atten3.shape[-1]))
        up_atten3 = self.deconv3_1.call(up_atten3)  # (20, 115, 135, 64)
        up_atten3 = self.deconv3_2.call(up_atten3)  # (20, 117, 137, 64)
        up_atten3 = self.deconv3_3.call(up_atten3)  # (20, 235, 275, 64)
        up_atten3 = tf.reshape(up_atten3, (-1, self.seq, up_atten3.shape[-3], up_atten3.shape[-2], up_atten3.shape[-1]))
        up_atten3 = self.deconv3_conv1(up_atten3)  # (1, 20, 234, 274, 64)
        up_atten3 = tf.concat([up_atten3, conv2], axis=-1)
        del conv2
        up_atten3 = self.deconv3_conv2(up_atten3)
        up_atten2 = self.atten2(up_atten3, up_atten3)  # (1, 20, 234, 274, 64)
        del up_atten3

        # upscale, attention 1
        up_atten2 = tf.reshape(up_atten2, (-1, up_atten2.shape[-3], up_atten2.shape[-2], up_atten2.shape[-1]))
        up_atten2 = self.deconv4_1.call(up_atten2)  # (20, 236, 276, 32)
        up_atten2 = self.deconv4_2.call(up_atten2)  # (20, 238, 278, 32)
        up_atten2 = self.deconv4_3.call(up_atten2)  # (20, 477, 557, 32)
        up_atten2 = tf.reshape(up_atten2, (-1, self.seq, up_atten2.shape[-3], up_atten2.shape[-2], up_atten2.shape[-1]))
        up_atten2 = self.deconv4_conv1(up_atten2)  # (1, 20, 476, 556, 32)
        up_atten2 = tf.concat([up_atten2, conv1], axis=-1)
        del conv2
        up_atten2 = self.deconv4_conv2(up_atten2)
        up_atten1 = self.atten1(up_atten2, up_atten2)  # (1, 20, 476, 556, 32)
        del up_atten2

        # convt, prepare to output
        output = tf.reshape(up_atten1, (-1, up_atten1.shape[-3], up_atten1.shape[-2], up_atten1.shape[-1]))
        output = self.deconv5_1.call(output)  # (20, 478, 558, 16)
        output = self.deconv5_2.call(output)  # (20, 480, 560, 8)
        output = self.deconv5_conv(output)  # (20, 480, 560, 1)
        output = tf.reshape(output, (-1, self.seq, output.shape[-3], output.shape[-2]))  # (1, 20, 480, 560)
        output = self.sigmoid(output)

        return output


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "/gpu:0"
    import numpy as np

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    inputs_x = np.random.random((1, 1, 480, 560, 1))
    inputs_x = tf.convert_to_tensor(inputs_x)
    # inputs = tf.keras.Input(shape=inputs_x.shape)

    model = Net(head_nums=1, seq=1)
    # model.build(input_shape=x.shape)
    output = model.call(inputs_x)
    print(output.shape)

    # # tf.debugging.set_log_device_placement(True)
    # net1 = deconv2d_bn(out_channels=16, strides=(1, 1))
    # net2 = deconv2d_bn(out_channels=8, strides=(1, 1))
    # # net3 = deconv2d_bn(out_channels=32, strides=(2, 2))
    #
    # x = tf.random.normal((1, 10, 476, 556, 32))
    # # c3 = tf.random.normal((1, 10, 476, 556, 32))
    # print('x', x.shape)
    # x = tf.reshape(x, (-1, x.shape[-3], x.shape[-2], x.shape[-1]))
    # new_x = net1.call(x)
    # print('convT1 x', new_x.shape)
    # del x
    # new_x = net2.call(new_x)
    # print('convT2 x', new_x.shape)
    # # new_x = net3.call(new_x)
    # # print('convT3 x', new_x.shape)
    #
    # conv = layers.Conv2D(filters=1, kernel_size=1)
    # new_x = conv(new_x)
    # print('final conv', new_x.shape)
    #
    # new_x = tf.reshape(new_x, (-1, 10, new_x.shape[-3], new_x.shape[-2]))
    # print('reshaped:', new_x.shape)
    #
    # # sg = keras.activations.sigmoid
    # sg = tf.nn.sigmoid
    # new_x = sg(new_x)
    # print('sigmoid', new_x.shape)
    #
    # # net4 = layers.Conv2D(filters=32, kernel_size=2)
    # # new_x = net4(new_x)
    # # print('conv4 x', new_x.shape)
    #
    # # added = tf.concat([new_x, c3], axis=-1)
    # # print('added x:', added.shape)
    #
    # # net5 = layers.Conv2D(filters=32, kernel_size=1)
    # # new_x = net5(added)
    # # print('conv5 x', new_x.shape)

    # bn1 = layers.BatchNormalization()
    # bnx = bn1(new_x)
    # del new_x
    # print('bn x', bnx.shape)
    # out = tf.nn.relu(bnx)
    # del bnx
    # print('relu', out.shape)
    # out = tf.reshape(out, (-1, out.shape[-3], out.shape[-2], out.shape[-1]))
    # pool = layers.AveragePooling2D(pool_size=(2, 2), input_shape=out.shape[2:-1])
    # op = pool(out)
    # del out
    # op = tf.reshape(op, (-1, 20, op.shape[-3], op.shape[-2], op.shape[-1]))
    # print('pool 2d', op.shape)
    # x = tf.random.normal((16, 20, 28, 28, 1024))
    # print(x.shape)
    # att = layers.MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(1, 2))
    # new_x = att(x, x)
    # print(new_x.shape)
