# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from semilearn.core.utils import get_net_builder, get_dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str, required=True)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='resnet8x4')
    parser.add_argument('--net_from_name', type=bool, default=False)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=int, default=0.875)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)

    args = parser.parse_args()
    
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['model']
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            key = '.'.join(key.split('.')[1:])
        # 添加 backbone 前缀
        if key.startswith('backbone.'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    args.save_dir = save_dir
    args.save_name = ''
    
    net = get_net_builder(args.net, args.net_from_name)(num_classes=args.num_classes)
    keys = net.load_state_dict(load_state_dict, strict=False)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    
    # specify these arguments manually 
    args.num_labels = 400
    args.ulb_num_labels = 49600
    args.lb_imb_ratio = 1
    args.ulb_imb_ratio = 1
    args.seed = 0
    args.epoch = 1
    args.num_train_iter = 1024
    dataset_dict = get_dataset(args, 'fixmatch', args.dataset, args.num_labels, args.num_classes, args.data_dir, False)
    eval_dset = dataset_dict['eval']
    eval_loader = DataLoader(eval_dset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=4)
    # 查看第一个batch 的类型
    # for data in eval_loader:
    #     print(data['y_lb'][:5])
    #     break

    acc = 0.0
    test_logits = []
    test_feats = []
    test_preds = []
    test_probs = []
    test_labels = []
    with torch.no_grad():
        for data in eval_loader:
            image = data['x_lb']
            target = data['y_lb']

            image = image.type(torch.FloatTensor).cuda()
            # feat = net(image, only_feat=True)
            # logit = net(feat, only_fc=True)
            outs_x = net(image)
            logit = outs_x['logits']
            feat = outs_x['feat'][-1]
            prob = logit.softmax(dim=-1)
            pred = prob.argmax(1)

            acc += pred.cpu().eq(target).numpy().sum()

            test_logits.append(logit.cpu().numpy())
            test_feats.append(feat.cpu().numpy())
            test_preds.append(pred.cpu().numpy())
            test_probs.append(prob.cpu().numpy())
            test_labels.append(target.cpu().numpy())
    test_logits = np.concatenate(test_logits)
    test_feats = np.concatenate(test_feats)
    test_preds = np.concatenate(test_preds)
    test_probs = np.concatenate(test_probs)
    test_labels = np.concatenate(test_labels)

    print(f"Test Accuracy: {acc/len(eval_dset)}")
    print(f"logit shape: {test_logits.shape}")
    # print(f"feat shape: {test_feats.shape}")

    # index = np.random.choice(len(eval_dset), 1000, replace=False)
    # test_feats = test_feats[index]
    # test_labels = test_labels[index]
    pca = PCA(n_components=50)
    pca_features = pca.fit_transform(test_logits) # (10000,50)

    tsne = TSNE(n_components=3, random_state=42, perplexity=40, max_iter=500)
    features_tsne = tsne.fit_transform(test_logits) # (10000,3)


    # plt.figure(figsize=(10, 8))
    # plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=test_labels, cmap='viridis', alpha=0.6)

    # # 添加颜色条
    # plt.colorbar()

    # # 添加标题和标签
    # plt.title('t-SNE Visualization')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')

    # # 显示图像
    # plt.show()
    # plt.savefig('zzz_tsne/' + args.net + '_' + args.load_path[21:-17] + '.png', dpi=300, bbox_inches='tight')

    # 创建3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 3D散点图
    scatter = ax.scatter(features_tsne[:, 0], 
                        features_tsne[:, 1], 
                        features_tsne[:, 2], 
                        c=test_labels, 
                        cmap='viridis', 
                        alpha=0.6)

    # 添加颜色条
    plt.colorbar(scatter)

    # 添加标题和标签
    ax.set_title('3D t-SNE Visualization')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')

    # 保存图像
    plt.savefig('zzz_tsne/' + args.net + '_' + args.load_path[21:-17] + '_3d.png', dpi=300, bbox_inches='tight')
    # plt.show()
