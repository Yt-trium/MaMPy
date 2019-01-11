import sys
import time
import matplotlib.pyplot as plt
from utils import image_read

file_path = sys.argv[1]
area_filter = int(sys.argv[2])

img1 = image_read(filename=file_path)
print("Resolution: {}".format(img1.shape[0] * img1.shape[1]))
print("{}x{}".format(img1.shape[0], img1.shape[1]))

flatten_image = img1.flatten()

# Original image
fig, (fig1, fig2, fig3, fig4) = plt.subplots(1, 4)
fig1.set_axis_off();
fig2.set_axis_off();
fig3.set_axis_off();
fig4.set_axis_off();
fig1.set_title("Original")
fig1.imshow(img1, cmap="gray")

# Algorithms
start = time.time()
(parents, s) = maxtree_berger(img1, connection8=True)
end = time.time()
attr = compute_attribute_area(s, parents, flatten_image)
out1 = direct_filter(s, parents, flatten_image, attr, area_filter)
out1 = np.reshape(out1, img1.shape)
fig2.set_title("Maxtree classic \n[{}s]".format(round(end - start, 2)))
fig2.imshow(out1, cmap="gray")

# Algorithms
start = time.time()
(parents, s) = maxtree_berger_rank(img1, connection8=True)
end = time.time()
attr = compute_attribute_area(s, parents, flatten_image)
out2 = direct_filter(s, parents, flatten_image, attr, area_filter)
out2 = np.reshape(out2, img1.shape)
fig3.set_title("Maxtree Rank \n[{}s]".format(round(end - start, 2)))
fig3.imshow(out2, cmap="gray")

# Algorithms
start = time.time()
(parents, s) = maxtree_union_find_level_compression(img1, connection8=True)
end = time.time()
attr = compute_attribute_area(s, parents, flatten_image)
out3 = direct_filter(s, parents, flatten_image, attr, area_filter)
out3 = np.reshape(out3, img1.shape)
fig4.set_title("Maxtree Union & Level compression \n[{}s]".format(round(end - start, 2)))
fig4.imshow(out3, cmap="gray")


print("Différences (en nombre de pixels):")
print("Algo 1 et 2: {}".format(np.count_nonzero(out1 == out2)))
print("Algo 1 et 3: {}".format(np.count_nonzero(out1 == out3)))
print("Algo 2 et 3: {}".format(np.count_nonzero(out2 == out3)))
print(out1)
print(out2)
print(out3)


print(np.sum(np.abs(out1 - out2)))
print(np.sum(np.abs(out1 - out3)))
print(np.sum(np.abs(out2 - out3)))
# Show plot
plt.show()