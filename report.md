## Task 1.1: Forward Pass

LLaVA consists of multiple parts: the vision encoder, projection layer, and language model. There are some difference between LLaVa in the original paper and LLaVa 1.5/HD, which will be pointed out.

### Input

In the original LLaVa, the images should be of resolution 224 X 224 for the vision encoder. In LLaVa 1.5, it was increased to allow images of resolution 336 x 336. At the time, the highest resolution the vision encoder could support was 336 x 336, however, one way they got around was to first partition the image into parts, feed those parts to the vision enconder independently. The output feature maps are stitched back together based on their relative position on the image and finally flattened. An important part is to include the necessary global context since each feature map doesn't "know" about the others. To do this, the image is downsampled to a lower resolution, fed to the vision encoder, and finally the output is concantenated with the rest of the feature maps.

### Vision Encoder

The vision encoder used for the original LLaVa is [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14). This was later swapped to CLIP ViT-L-336px in version 1.5 to support resolutions up to 336 x 336. The vision encoder is frozen during training.

CLIP is a contrastive learning model. During pretraining, it takes in image-text pairs as inputs. The text is typically a description or a caption of the image. CLIP contains ViT and a transformer. Every image and text is fed into ViT and transformer, respectively, generating embeddings for both. It's important to note that both embeddings are in the same embedding space. This allows the model to compare the embddings of the images and text using cosine similary. The model learns by maximizing the cosine simiarity of true pairs embeddings while minimizing the cosine simiarity of negative pairs. 

### Projection Layer

In verision 1, a liner layer projects the embeddings from CLIP to the same embedding space as the transformer's embedding. In version 1.5, an MLP is used.

### Lnaguage Model Input

The languag model takes the projected image embeddings from the projection layer and the embeddings from the transforer.
