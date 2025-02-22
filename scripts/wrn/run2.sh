# python train.py --c config/OSKD_cv/dkd/dkd_cifar100_with_tiny_imagenet_50000_wrn_40_1_wrn_16_1_0.yaml
# python train.py --c config/OSKD_cv/srd/srd_cifar100_with_tiny_imagenet_50000_wrn_40_1_wrn_16_1_0.yaml 
# python train.py --c config/OSKD_cv/pad/pad_cifar100_with_tiny_imagenet_50000_wrn_40_1_wrn_16_1_0.yaml 

python train.py --c config/OSSL_cv/supervised/supervised_cifar100_with_tiny_imagenet_50000_wrn_16_1_0.yaml 
python train.py --c config/OSSL_cv/supervised/supervised_cifar100_with_tiny_imagenet_50000_shuffleV1_0.yaml
python train.py --c config/OSSL_cv/supervised/supervised_cifar100_with_tiny_imagenet_50000_vgg8_0.yaml 

