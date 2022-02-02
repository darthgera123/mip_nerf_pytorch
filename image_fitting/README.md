# Image Fitting
In this exercise, we go through latest techniques to fit an image to a neural network. There has been lot of works in the past 2 years in this domain and we try to understand and go through it. Experiment results can be found [here](https://wandb.ai/erenyeager/image_fit?workspace=user-erenyeager) 

# Problem
Given an image coordinate (x,y), we want to guess the color of that pixel (r,g,b). We normalize the images. The image coords range is `(x,y) => [0,1]`. Similarly, the range of pixel colors are `(rgb) => [0,1]`. For training, we take every alternate pixel as input. We predict the entire image. Checkout `dataset.py`.

# Solution
## Baseline
Our input to the network is 2 dimensional (x,y). The network architecture is as follows:
+ 3 linear layers + ReLU activation
+ 1 output linear layer + Sigmoid activation

We observe that the output is horrible and the network isnt able to learn any high frequencies at all. We now discuss various strategies to counter this. PSNR : 17.448 dB
## Change to input

### Positional Embedding
We transform the input `(x,y) =>(x,sin(x),cos(x),sin(2x),cos(2x)....,sin(2**Lx),cos(2**Lx),y,sin(y),...,sin(2**Ly),cos(2**Ly)`. The dimensionality of the input changes from 2 to `2*(2*L)+2` where L is a hyperparameter that controls how much high frequency data we want to encode. This is called positional embedding. It helps the network learn the high frequencies and ensures that it captures the fine details with ease. PSNR : 24.74 dB

### Fourier Features and Kernels
Now we transform the input as `(x,y) =>(Bx,sin(Bx),cos(Bx),sin(2Bx),cos(2Bx)....,sin(2**LBx),cos(2**LBx),By,sin(By),...,sin(2**LBy),cos(2**LBy)`. B is a tall matrix consisting of randomly sampled points from a `N ~(0,scale)`. This needs to be initialized once. We experiment with B's length as `[256,512,1024]`.Best results were obtained at 1024 and `scale =1`. PSNR: 31.96 dB. The quality of image is a function of the scale aka the width of the gaussian and the length of the matrix.

#### Theory
[This paper](https://ar5iv.org/html/2006.10739) introduces the concepts. If we have a MLP with a very wide layer, then [Neural Tangent Kernel](https://ar5iv.org/html/2006.10739) theory says that `NTK ~ Kernel Regression`. [Kernel Regression] (https://towardsdatascience.com/kernel-regression-made-easy-to-understand-86caf2d2b844) is a method for estimating a non linear curve fitting the points. We assume the Kernel to be gaussian and the width of the Kernel defines if we are underfitting or overfitting the curve. The B matrix is the kernel and the sin(Bx),cos(Bx) are the fourier features which help us estimate better. Positional Embeddings are a special case of this. [Checkout this video](https://www.youtube.com/watch?v=iKyIJ_EtSkw)

## Change to network

### SIREN
Instead of ReLU activations, we now work with Sine as our activation function. This gives us lot of flexibilty as well as power. However we need to ensure that our initilization is done right since sine is a periodic function and can get stuck at local minima. [This blog](https://medium.com/@sallyrobotics.blog/sirens-implicit-neural-representations-with-periodic-activation-functions-f425c7f710fa) explains in detail what all we can achieve with SIREN. [Code](https://github.com/vsitzmann/siren) and [Colab Notebook](https://colab.research.google.com/drive/1PhQmqMydYFRz3FNr9onCdANQfKK9pYqM#scrollTo=tzQ2fnQOVRSv)
