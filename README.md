# ATUNET
My net for weather forecast in TIANCHI test. <p>
The main idea is use UNET and Attention Mechanism.<p>
## Steps
1. downscale for 5 times.
2. use multi-head self-attention at the bottom.
3. upscale for 5 times by using Convolution Transpose, 
residual connection along with Convolution will be used for crop 
after each upscaling, as well as attention if OOM doesn't appear.
4. each attention layer is used for time sequence forecast.
## some infos
The shape of the input data will be `(batch_size, seq, h, w)`, where `seq` denotes the sequence length(here `seq=20`).<p>
Then it will be reshaped to `(batch_size, seq, h, w, channels)` by adding a dim at the last axis.<p>
So the shape of the actual data received by the `model` is `(batch_size, seq, h, w, 1)`.<p>
And the shape of th output is `(batch_size, seq, h, w)` <p>
Use early-stop and AdamWarmup.<p>
Use dice_coef as the loss function. <p>
## TODO
1. examzine train.py
2. predict.py
3. metric.py
4. save the results as image
