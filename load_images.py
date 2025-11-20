# import lvpyio as lv

# path = r"images\U1_35_ER1_0,85_LBO_3\Camera1.cine"
# lv.is_multiset(path)
# s = lv.read_set(path)

import cinereader as cr
import imageio.v2 as imageio

cine_path = r"images\U1_35_ER1_0,85_LBO_3\Camera1.cine"

# Read just metadata
meta = cr.read_metadata(cine_path)

# Read all frames + timestamps
meta, images, timestamps = cr.read(
    cine_path,
    start=meta.FirstImageNo,
    count=meta.ImageCount   # or meta.ImageCount - 1 depending on version
)

first_frame = images[1000]  # numpy array
print(first_frame.shape, first_frame.dtype)

imageio.imwrite("frame0.png", first_frame)
