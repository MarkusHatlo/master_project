import numpy as np
import lvpyio as lv
import imageio
import matplotlib.pyplot as plt


path = r"D:\Qian_Markus_August_2025\Mean_image\U1_20_ER1_0,72_D_88mm_H_260mm_Mean_im.set"
lv.is_multiset(path)

# # Indices you want to plot
# indices = [1000 * i for i in range(1, 10)]  # 1000,2000,...,9000

# fig, axes = plt.subplots(3, 3, figsize=(7, 7))

# for ax, idx in zip(axes.flatten(), indices):
#     s = lv.read_set(path)[idx]
#     img_array = s.as_masked_array()

#     im = ax.imshow(img_array, cmap='hot', origin='lower')
#     ax.set_title(f"Index {idx}")
#     ax.axis('off')

# # Add one shared colorbar
# fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)

# # plt.tight_layout()
# plt.show()

def make_mp4():
    writer = imageio.get_writer("output.mp4", fps=10)

    b = lv.read_set(path)  # read whole multiset

    for i in range(len(b)):
        frame = b[i].as_masked_array()
        writer.append_data(frame.astype(np.uint8))   # adjust uint8 if needed

    writer.close()

S = lv.read_set(path)
N = len(S)

# Read the first frame to get the shape
first = S[0].as_masked_array()
mean_img = np.zeros_like(first, dtype=float)
count = np.zeros_like(first, dtype=float)  # for masked pixels

# Accumulate
for i in range(N):
    print('idx: ', i)
    frame = S[i].as_masked_array()
    data = np.nan_to_num(frame)            # replace masked with 0 temporarily
    mask = ~np.isnan(frame)                # True where real data exists

    mean_img += data
    count += mask.astype(float)

mean_img = np.where(count > 0, mean_img / count, np.nan)

# Plot
plt.imshow(mean_img, cmap='hot', origin='upper')
plt.colorbar()
plt.title("Mean Image")
plt.show()
