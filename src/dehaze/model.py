"""Encoder-decoder (U-Net style) dehazing network.

* 3 downsampling stages (stride-2 convs) to a residual bottleneck, then 3 upsampling
  stages with skip connections, so fine texture from the encoder survives.
* Predicts a residual that is added to the hazy input (global skip): the network only
  has to learn what haze changed, which trains faster and keeps colours stable.
* No batch norm: statistics would differ between training patches and tiled
  full-size inference, which shows up as seams between tiles.
* Fully convolutional with a dynamic input size (multiples of 8), so the same model
  runs on 256 px patches or larger tiles, in Python and in the browser (ONNX).
"""


def build_model(tf, base=32, n_res=4):
    L = tf.keras.layers

    def conv(x, f, k=3, s=1):
        return L.LeakyReLU(0.1)(L.Conv2D(f, k, strides=s, padding="same")(x))

    def res_block(x, f):
        y = conv(x, f)
        y = L.Conv2D(f, 3, padding="same")(y)
        return L.Add()([x, y])

    inp = L.Input((None, None, 3), name="hazy")
    e1 = conv(conv(inp, base), base)                 # 1/1
    e2 = conv(conv(e1, base * 2, s=2), base * 2)     # 1/2
    e3 = conv(conv(e2, base * 4, s=2), base * 4)     # 1/4
    b = conv(e3, base * 8, s=2)                      # 1/8
    for _ in range(n_res):
        b = res_block(b, base * 8)

    def up(x, skip, f):
        x = L.Conv2DTranspose(f, 4, strides=2, padding="same")(x)
        x = L.Concatenate()([x, skip])
        return conv(conv(x, f), f)

    d3 = up(b, e3, base * 4)
    d2 = up(d3, e2, base * 2)
    d1 = up(d2, e1, base)
    residual = L.Conv2D(3, 3, padding="same", dtype="float32", name="residual")(d1)
    out = L.Add(dtype="float32", name="dehazed")([L.Activation("linear", dtype="float32")(inp), residual])
    return tf.keras.Model(inp, out, name="dehaze_unet")
