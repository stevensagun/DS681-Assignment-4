## Task 1.1: Forward Pass

LLaVA consists of multiple parts: the vision encoder, projection layer, and language model. There are some difference between LLaVa in the original paper and LLaVa 1.5/HD, which will be pointed out.

### Input

In the original LLaVa, the images should be of resolution 224 X 224 for the ViT. In LLaVa 1.5, it was increased to allow images of resolution 336 x 336. At the time, the highest resolution the vision encoder could support was 336 x 336, however, one way they got around was to first partition the image into parts, feed those parts to the vision enconder independently. The output feature maps are stitched back together based on their relative position on the image and finally flattened. An important part is to include the necessary global context since each feature map doesn't "know" about the others. To do this, the image is downsampled to a lower resolution, fed to the vision encoder, and finally the output is concantenated with the rest of the feature maps.

### Vision Encoder
