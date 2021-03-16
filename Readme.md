# Deep Learning Method in Image Compression

In this project, we use Autoencoder and GAN to tackle this problem. We train our models on **DIV2K** dataset and evaluate the performance on JPEG AI public test dataset based on PSNR and SSIM.

# Quickstart

```sh
python3 train.py --dataDir path_to_your_training_data 
```

```sh
python3 test.py --dataDir path_to_your_testing_data --encoderWeight ./checkpoint/encoder.pt --decoderWeight ./checkpoint/decoder.pt
```

# Evaluation
|     Original      |                 Reconstructed                  |
| :---------------: | :--------------------------------------------: |
| ![](assets/1.png) |              ![](assets/1_re.png)              |
|      5.6 MB       | Compressed to 774 KB (PSNR: 72.70, SSIM: 0.59) |
| ![](assets/2.png) |              ![](assets/2_re.png)              |
|      4.7 MB       | Compressed to 627 KB (PSNR: 70.67, SSIM: 0.66) |
| ![](assets/3.png) |              ![](assets/3_re.png)              |
|      6.2 MB       | Compressed to 887 KB (PSNR: 77.65, SSIM: 0.68) |
| ![](assets/4.png) |              ![](assets/4_re.png)              |
|      12.9 MB      | Compressed to 2.3 MB (PSNR: 74.99, SSIM: 0.80) |


