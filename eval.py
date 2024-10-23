# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import os
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from torch.utils.data import DataLoader
from semilearn.core.utils import get_net_builder, get_dataset



def eval(args, net, eval_loader):
    acc = 0.0
    # test_feats = []
    # test_preds = []
    # test_probs = []
    # test_labels = []
    # input = torch.randn(1, 3, 32, 32).cuda()
    # logit = net(input)['logits']
    # print("logit", logit.shape)

    seen_classes = np.random.RandomState(args.seed).choice(args.num_classes, args.ID_classes, replace=False)
    seen_classes = np.sort(seen_classes)
    # print("target shape", )
    print("seen_classes", seen_classes)
    with torch.no_grad():
        for data in eval_loader:
            image = data['x_lb'].cuda()
            target = data['y_lb'].cuda()

            image = image.type(torch.FloatTensor).cuda()
            feat = net(image, only_feat=True)[-1]
            logit = net(feat, only_fc=True)

            # print(f"chaged logit shape: {logit.shape}")
            # print(f"target shape: {target.shape}")
            # print(f"target[:5]: {target[:5]}")
            # print(type(target), target.is_cuda)
            
            if logit.shape[1] != args.ID_classes:
                logit = logit[:, seen_classes]
                # print(f"logit shape: {logit.shape}")
            prob = logit.softmax(dim=-1)
            pred = prob.argmax(1).cpu()
            pred = [seen_classes[i] for i in pred]
            pred = torch.tensor(pred)
            # print(type(pred))
            # print(f"pred[:5]: {pred[:5]}")

            acc += target.cpu().eq(pred).sum().item()

    #         test_feats.append(feat.cpu().numpy())
    #         test_preds.append(pred.cpu().numpy())
    #         test_probs.append(prob.cpu().numpy())
    #         test_labels.append(target.cpu().numpy())
    # test_feats = np.concatenate(test_feats)
    # test_preds = np.concatenate(test_preds)
    # test_probs = np.concatenate(test_probs)
    # test_labels = np.concatenate(test_labels)

    print(f"Test Accuracy: {acc/len(eval_loader.dataset)}")
    # return test_feats, test_preds, test_probs, test_labels

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str, default='saved_models/pre_mod/supervised_tiny_imagenet_with_cifar_wide_resnet50_2_200_500_0_500_0/model_best.pth')
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='wide_resnet50_2')  
    parser.add_argument('--net_from_name', type=bool, default=False)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='tiny_imagenet_with_cifar')
    parser.add_argument('--num_classes', type=int, default=200)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=int, default=0.875)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)

    parser.add_argument('--ID_classes', type=int, default=50)
    parser.add_argument('--ID_labels_per_class', type=int, default=500)
    parser.add_argument('--OOD_classes', type=int, default=0)
    parser.add_argument('--OOD_labels_per_class', type=int, default=500)

    args = parser.parse_args()


    eval_name = args.load_path.split('/')[-2]
    print(eval_name)
    eval_name = eval_name.split('_')

    # args.ID_classes = int(eval_name[3])
    # args.ID_labels_per_class = int(eval_name[4])
    # args.OOD_classes = int(eval_name[5])
    # args.OOD_labels_per_class = int(eval_name[6])

    args.seed = int(eval_name[-1])
    
    checkpoint_path = os.path.join(args.load_path)

    checkpoint = torch.load(checkpoint_path)

    load_model = checkpoint['ema_model']

    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item

    # save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    
    # args.save_dir = save_dir
    # args.save_name = ''
    
    net = get_net_builder(args, args.net, args.net_from_name)(num_classes=args.num_classes)
    keys = net.load_state_dict(load_state_dict)
    
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    # specify these arguments manually 
    args.num_labels = 400
    args.lb_imb_ratio = 1
    args.ulb_imb_ratio = 1
    args.seed = 0
    args.epoch = 1
    
    dataset_dict = get_dataset(args, 'supervised')
    eval_dset = dataset_dict['eval']
    print(f"eval_dset: {len(eval_dset)}")

    args.num_train_iter = len(eval_dset) // args.batch_size
    print(f"num_train_iter: {args.num_train_iter}")
    eval_loader = DataLoader(eval_dset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=8)
    # 查看第一个batch 的类型
    # for data in eval_loader:
    #     print(data['y_lb'][:5])
    #     break
    print("load_path", args.load_path)

    eval(args, net, eval_loader)



