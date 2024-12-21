import os

def create_configuration(cfg, cfg_file):
    cfg["save_name"] = "{alg}_{dataset}_{net}_{ID_class}_{ID_labels_per_class}_{OOD_class}_{OOD_labels_per_class}_{seed}".format(
        alg=cfg["algorithm"],
        dataset=cfg["dataset"],
        net=cfg["net"],
        ID_class=cfg["ID_classes"],
        ID_labels_per_class=cfg["ID_labels_per_class"],
        OOD_class=cfg["OOD_classes"],
        OOD_labels_per_class=cfg["OOD_labels_per_class"],
        seed=cfg["seed"]
    )
    # print(cfg["save_name"])
    # resume
    cfg["resume"] = True
    cfg["load_path"] = "{}/{}/latest_model.pth".format(
        cfg["save_dir"], cfg["save_name"]
    )

    alg_file = cfg_file + cfg["algorithm"] + "/"
    if not os.path.exists(alg_file):
        os.mkdir(alg_file)

    # print(alg_file + cfg["save_name"] + ".yaml")
    with open(alg_file + cfg["save_name"] + ".yaml", "w", encoding="utf-8") as w:
        lines = []
        for k, v in cfg.items():
            line = str(k) + ": " + str(v)
            lines.append(line)
        for line in lines:
            w.writelines(line)
            w.write("\n")

def create_pre_mod_config(
    alg,
    seed,
    dataset,
    num_ulb,
    net,
    ID_classes,
    ID_labels_per_class,
    OOD_classes,
    OOD_labels_per_class,
    img_size,
    crop_ratio,
    port,
    lr,
    lr_decay_epochs,
    weight_decay,
    layer_decay=1,
    warmup=5,
    amp=False
):
    cfg = {}
    cfg["algorithm"] = alg

    # save config
    cfg["save_dir"] = "./saved_models/pre_mod/"
    cfg["save_name"] = None
    cfg["resume"] = False
    cfg["load_path"] = None
    cfg["overwrite"] = True
    cfg["use_tensorboard"] = True
    cfg["use_wandb"] = False
    cfg["use_aim"] = False

    cfg["epoch"] = 200
    cfg["num_lb"] = ID_classes * ID_labels_per_class
    cfg["num_ulb"] = num_ulb

    cfg["batch_size"] = 128
    cfg["eval_batch_size"] = 256

    cfg["iter_per_epoch"] = cfg["num_lb"] // cfg["batch_size"] 
    cfg["num_train_iter"] = cfg["epoch"] * cfg["iter_per_epoch"]
    cfg["num_log_iter"] = cfg["iter_per_epoch"] 
    cfg["num_eval_iter"] = cfg["iter_per_epoch"] * 2
    

    cfg["num_warmup_iter"] = int(cfg["iter_per_epoch"] * warmup)
    
    cfg["ID_classes"] = ID_classes
    cfg["ID_labels_per_class"] = ID_labels_per_class
    if OOD_classes is not None:
        cfg["OOD_classes"] = OOD_classes
        cfg["OOD_labels_per_class"] = OOD_labels_per_class

    cfg["uratio"] = 1
    cfg["ema_m"] = 0.0

    if alg == "fixmatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        if dataset == "imagenet":
            cfg["ulb_loss_ratio"] = 10.0
            cfg["p_cutoff"] = 0.7

    elif alg == "pseudolabel":
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg["unsup_warm_up"] = 0.4

    elif alg == "meanteacher":
        cfg["ulb_loss_ratio"] = 50
        cfg["unsup_warm_up"] = 0.4
        cfg["ema_m"] = 0.999

    elif alg == "dash":
        cfg["gamma"] = 1.27
        cfg["C"] = 1.0001
        cfg["rho_min"] = 0.05
        cfg["num_wu_iter"] = 2048
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0

    elif alg == "iomatch":
        cfg["ema_m"] = 0.999

    cfg["img_size"] = img_size
    cfg["crop_ratio"] = crop_ratio

    # optim config
    cfg["optim"] = "SGD"
    cfg["lr"] = lr
    cfg["layer_decay"] = layer_decay
    cfg["lr_decay_epochs"] = lr_decay_epochs
    cfg["momentum"] = 0.9
    cfg["weight_decay"] = weight_decay
    cfg["amp"] = amp
    cfg["clip"] = 0.0
    cfg["use_cat"] = True

    # net config
    cfg["net"] = net
    cfg["net_from_name"] = False

    # data config
    cfg["data_dir"] = "./data"
    cfg["dataset"] = dataset
    cfg["train_sampler"] = "RandomSampler"
    cfg["num_workers"] = 16

    # basic config
    cfg["seed"] = seed

    # distributed config
    cfg["world_size"] = 1
    cfg["rank"] = 0
    cfg["multiprocessing_distributed"] = True
    cfg["dist_url"] = "tcp://127.0.0.1:" + str(port)
    cfg["dist_backend"] = "nccl"
    cfg["gpu"] = None

    # other config
    cfg["overwrite"] = True
    cfg["use_pretrain"] = True
    cfg["pretrain_path"] = "./pretrained_models/{}".format(net)
    # cfg["use_timm"] = True

    return cfg

def exp_pre_mod(ID_labels_per_class, ID_classes, OOD_classes):
    config_file = r"./config/pre_mod/"
    save_path = r"./saved_models/pre_mod"

    if not os.path.exists(config_file):
        os.mkdir(config_file)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    algs = [
        # "fixmatch",
        # "pseudolabel",
        # "meanteacher",
        # "dash",
        "supervised"
        # "iomatch",
    ]

    nets = [
        # "resnet8x4",
        # "resnet32x4",
        # "wrn_16_1",
        # "wrn_40_1",
        # "shuffleV1",
        # "vgg8",
        # "vgg13",
        "wide_resnet50_2",
        "resnet34"
    ]

    datasets = [
        "tiny_imagenet_with_cifar"
    ]
    seeds = [0, 1, 2, 3, 4, 5]

    dist_port = range(10001, 25120, 1)
    count = 0

    weight_decay = 5e-4
    lr = 5e-3
    lr_decay_epochs = "100,150,180"
    layer_decay = 1.0
    warmup = 0
    amp = False
    img_size = 64

    for alg in algs:
        for dataset in datasets:
            for net in nets:
                for seed in seeds:
                    # changes for each dataset
                    if dataset == "cifar100_with_tin":
                        id_classes = ID_classes[0]
                        id_labels_per_class = ID_labels_per_class[0]
                        ood_classes = OOD_classes[0]
                        ood_labels_per_class = 500
                        num_ulb = id_classes * id_labels_per_class + ood_classes * ood_labels_per_class
                        crop_ratio = 0.875
                    else:
                        id_classes = ID_classes[1]
                        id_labels_per_class = ID_labels_per_class[1]
                        ood_classes = OOD_classes[0]
                        ood_labels_per_class = 500
                        num_ulb = id_classes * id_labels_per_class + ood_classes * ood_labels_per_class
                        crop_ratio = 0.875

                    port = dist_port[count]
                    cfg = create_pre_mod_config(
                        alg,
                        seed,
                        dataset,
                        num_ulb,
                        net,
                        id_classes,
                        id_labels_per_class,
                        ood_classes,
                        ood_labels_per_class,   
                        img_size,
                        crop_ratio,
                        port,
                        lr,
                        lr_decay_epochs,
                        weight_decay,
                        layer_decay,
                        warmup,
                        amp,
                    )
                    count += 1
                    create_configuration(cfg, config_file)

if __name__ == "__main__":
    if not os.path.exists("./saved_models/pre_mod/"):
        os.makedirs("./saved_models/pre_mod/", exist_ok=True)
    if not os.path.exists("./config/pre_mod/"):
        os.makedirs("./config/pre_mod/", exist_ok=True)
    ID_labels_per_class = {"full": [500,500]}
    ID_classes = {"m": [100, 200]}
    OOD_classes = {"empty":[0]}

    for i in ID_labels_per_class:
        for j in ID_classes:
            for k in OOD_classes:
                exp_pre_mod(ID_labels_per_class=ID_labels_per_class[i],
                    ID_classes=ID_classes[j],
                    OOD_classes=OOD_classes[k])


