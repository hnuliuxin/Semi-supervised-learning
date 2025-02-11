# python train.py --c config/ossl_cv/supervised/supervised_cinic10_resnet8x4_5_9000_5_9000_0.yaml

# python train.py --c config/ossl_cv/supervised/supervised_cinic10_resnet32x4_3_9000_7_9000_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cinic10_resnet32x4_5_9000_5_9000_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cinic10_resnet32x4_7_9000_3_9000_0.yaml

# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet32x4_100_500_200_500_0.yaml



# python train.py --c config/ossl_cv/supervised/supervised_cinic10_resnet32x4_3_7200_7_9000_0.yaml
# # python train.py --c config/ossl_cv/supervised/supervised_cinic10_resnet32x4_5_7200_5_9000_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cinic10_resnet32x4_7_7200_3_9000_0.yaml

# python train.py --c config/ossl_cv/supervised/supervised_cinic10_resnet32x4_3_5400_7_9000_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cinic10_resnet32x4_5_5400_5_9000_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cinic10_resnet32x4_7_5400_3_9000_0.yaml


# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet18_100_500_0_500_1.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet18_100_500_0_500_2.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_2_100_500_0_500_0.yaml

# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_1_100_500_0_500_0.yaml


# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_1_100_500_0_500_1.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_1_100_500_0_500_2.yaml

# python train.py --c config/pre_mod/supervised/supervised_tiny_imagenet_with_cifar_resnet34_200_500_0_500_0.yaml


# 0的学习率为0.05，同时图片大小为32x32  所以效果较差
# python train.py --c config/pre_mod/supervised/supervised_tiny_imagenet_with_cifar_wide_resnet50_2_200_500_0_500_0.yaml

# 1 2的学习率为0.005， 图片大小都改成了64*64    同时1使用了只解冻了fc层
# 1花了5小时20分钟
# python train.py --c config/pre_mod/supervised/supervised_tiny_imagenet_with_cifar_wide_resnet50_2_200_500_0_500_1.yaml
# 模型所有参数参与微调
# python train.py --c config/pre_mod/supervised/supervised_tiny_imagenet_with_cifar_wide_resnet50_2_200_500_0_500_2.yaml
# 改动batch_size为128
# python train.py --c config/pre_mod/supervised/supervised_tiny_imagenet_with_cifar_wide_resnet50_2_200_500_0_500_3.yaml

# python train.py --c config/pre_mod/supervised/supervised_tiny_imagenet_with_cifar_wide_resnet50_2_200_500_0_500_5.yaml




###########################################

python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet8x4_100_200_50_400_0.yaml
python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet8x4_100_200_100_300_0.yaml
python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet8x4_100_200_150_240_0.yaml
python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet8x4_100_200_200_200_0.yaml


python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet32x4_100_200_50_400_0.yaml
python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet32x4_100_200_100_300_0.yaml
python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet32x4_100_200_150_240_0.yaml
python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet32x4_100_200_200_200_0.yaml

python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_resnet32x4_resnet8x4_100_200_50_400_0.yaml
python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_resnet32x4_resnet8x4_100_200_100_300_0.yaml
python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_resnet32x4_resnet8x4_100_200_150_240_0.yaml
python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_resnet32x4_resnet8x4_100_200_200_200_0.yaml

############################################

# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet8x4_100_250_50_500_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet8x4_100_250_100_375_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet8x4_100_250_150_300_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet8x4_100_250_200_250_0.yaml


# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet32x4_100_250_50_500_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet32x4_100_250_100_375_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet32x4_100_250_150_300_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_resnet32x4_100_250_200_250_0.yaml

# python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_resnet32x4_resnet8x4_100_250_50_500_0.yaml
# python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_resnet32x4_resnet8x4_100_250_100_375_0.yaml
# python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_resnet32x4_resnet8x4_100_250_150_300_0.yaml
# python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_resnet32x4_resnet8x4_100_250_200_250_0.yaml

############################################

# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_1_100_200_50_400_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_1_100_200_100_300_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_1_100_200_150_240_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_1_100_200_200_200_0.yaml


# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_2_100_200_50_400_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_2_100_200_100_300_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_2_100_200_150_240_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_2_100_200_200_200_0.yaml

# python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_wrn_40_2_wrn_40_1_100_200_50_400_0.yaml
# python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_wrn_40_2_wrn_40_1_100_200_100_300_0.yaml
# python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_wrn_40_2_wrn_40_1_100_200_150_240_0.yaml
# python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_wrn_40_2_wrn_40_1_100_200_200_200_0.yaml

# ############################################


# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_1_100_250_50_500_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_1_100_250_100_375_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_1_100_250_150_300_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_1_100_250_200_250_0.yaml


# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_2_100_250_50_500_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_2_100_250_100_375_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_2_100_250_150_300_0.yaml
# python train.py --c config/ossl_cv/supervised/supervised_cifar100_with_tin_wrn_40_2_100_250_200_250_0.yaml

# python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_wrn_40_2_wrn_40_1_100_250_50_500_0.yaml
# python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_wrn_40_2_wrn_40_1_100_250_100_375_0.yaml
# python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_wrn_40_2_wrn_40_1_100_250_150_300_0.yaml
# python train.py --c config/oskd_cv/kd/kd_cifar100_with_tin_wrn_40_2_wrn_40_1_100_250_200_250_0.yaml



# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_resnet8x4_100_200_50_400_0.yaml 
# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_resnet8x4_100_200_100_300_0.yaml 
# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_resnet8x4_100_200_150_240_0.yaml 
# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_resnet8x4_100_200_200_200_0.yaml 

# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_resnet8x4_100_250_50_500_0.yaml 
# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_resnet8x4_100_250_100_375_0.yaml 
# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_resnet8x4_100_250_150_300_0.yaml 
# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_resnet8x4_100_250_200_250.yaml 

# ## wrn40_1
# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_wrn_40_1_100_200_50_400_0.yaml 
# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_wrn_40_1_100_200_100_300_0.yaml 
# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_wrn_40_1_100_200_150_240_0.yaml 
# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_wrn_40_1_100_200_200_200_0.yaml 

# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_wrn_40_1_100_250_50_500_0.yaml 
# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_wrn_40_1_100_250_100_375_0.yaml 
# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_wrn_40_1_100_250_150_300_0.yaml 
# python train.py --c config/ossl_cv/iomatch/iomatch_cifar100_with_tin_wrn_40_1_100_250_200_250.yaml 
