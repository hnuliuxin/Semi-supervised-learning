import os

def create_configuration(cfg, cfg_file):
    cfg["save_name"] = "{alg}_{dataset}_{num_lb}_{net1}_{net}_{seed}".format(
        alg=cfg["algorithm"],
        dataset=cfg["dataset"],
        num_lb=cfg["num_labels"],
        net1=cfg["net_teacher"],
        net=cfg["net"],
        seed=cfg["seed"],
    )
    # resume
    # cfg["resume"] = True
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

def create_ossl_cv_config(
    alg,
    seed,
    dataset,
    net,
    num_classes,
    num_labels,
    img_size,
    crop_ratio,
    port,
    lr,
    weight_decay,
    layer_decay,
    warmup=5,
    amp=False,
):
    cfg = {}
    cfg["algorithm"] = alg

    # save config
    cfg["save_dir"] = "./saved_models/OSKD_cv/"
    cfg["save_name"] = None
    cfg["resume"] = False
    cfg["load_path"] = None
    cfg["overwrite"] = True
    cfg["use_tensorboard"] = True
    cfg["use_wandb"] = False
    cfg["use_aim"] = False

    if dataset == "imagenet":
        cfg["epoch"] = 500
        cfg["num_train_iter"] = 1024 * 500
        cfg["num_log_iter"] = 256
        cfg["num_eval_iter"] = 5120
        cfg["batch_size"] = 256
        cfg["eval_batch_size"] = 512
    else:
        cfg["epoch"] = 200
        cfg["num_train_iter"] = 1024 * 200
        cfg["num_log_iter"] = 256
        cfg["num_eval_iter"] = 2048
        cfg["batch_size"] = 64
        cfg["eval_batch_size"] = 128

    cfg["num_warmup_iter"] = int(1024 * warmup)
    cfg["num_labels"] = num_labels

    cfg["uratio"] = 1
    cfg["ema_m"] = 0.0

    # 算法的配置
    cfg["T"] = 1
    cfg["gamma"] = 1
    cfg["alpha"] = 1
    cfg["beta"] = 1

    if alg == "srd":
        cfg["criterion_kd_weight"] = 10
    elif alg == "dtkd":
        cfg["T"] = 0.1
        cfg["p_cutoff"] = 0.95

    # ulb_loss_ratio没有设置，因为每个算法的都不同。默认为1

    cfg["img_size"] = img_size
    cfg["crop_ratio"] = crop_ratio

    # optim config
    cfg["optim"] = "AdamW"
    cfg["lr"] = lr
    cfg["layer_decay"] = layer_decay
    cfg["momentum"] = 0.9
    cfg["weight_decay"] = weight_decay
    cfg["amp"] = amp
    cfg["clip"] = 0.0
    cfg["use_cat"] = True

    # net config
    cfg["net_teacher"] = net[0]
    path = os.path.join("./saved_models/OSSL_cv/")
    path = os.path.join(path, "supervised_{dataset}_{num_lb}_{net}_0".format(
        dataset=dataset,
        num_lb=num_labels,
        net=net[0],

    ))
    cfg["net_teacher_path"] = os.path.join(path, "model_best.pth")
    cfg["net"] = net[1]
    cfg["net_from_name"] = False

    # data config
    cfg["data_dir"] = "./data"
    cfg["dataset"] = dataset
    cfg["train_sampler"] = "RandomSampler"
    cfg["num_classes"] = num_classes
    cfg["num_workers"] = 8

    # basic config
    cfg["seed"] = seed

    # distributed config
    cfg["world_size"] = 1
    cfg["rank"] = 0
    cfg["multiprocessing_distributed"] = False
    cfg["dist_url"] = "tcp://127.0.0.1:" + str(port)
    cfg["dist_backend"] = "nccl"
    cfg["gpu"] = 0

    # other config
    cfg["overwrite"] = True
    cfg["use_pretrain"] = False

    return cfg

def exp_OSKD_cv(label_amount):
    config_file = r"./config/OSKD_cv/"
    save_path = r"./saved_models/OSKD_cv"

    if not os.path.exists(config_file):
        os.mkdir(config_file)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    algs = [
        "kd",
        "dkd",
        "srd",
        "fitnet",
        "dtkd",
        "pad",
        "my"
    ]

    nets = [
        ["resnet32x4", "resnet8x4"],
        ["resnet32x4", "shuffleV1"],
        # ["wrn_40_2", "wrn_40_1"],
        # ["wrn_40_4", "wrn_16_2"],
        # ["wrn_40_4", "wrn_16_4"],
        # ["resnet34", "resnet10"],
        # ["resnet50", "resnet18"],
        # ["resnet34", "wrn_16_2"],
        ["vgg13", "vgg8"],
        ["wrn_40_1", "wrn_16_1"]
    ]

    datasets = [
        "cifar100_with_tiny_imagenet",
        # "cifar100_with_places365", 
        # "cifar100", 
        # "tiny_imagenet", 
        # "cifar100_and_tiny_imagenet", 
        # "cifar100_and_places365"
    ]
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]

    dist_port = range(10001, 15120, 1)
    count = 0

    weight_decay = 5e-4
    # lr = 5e-5
    warmup = 0
    amp = False

    for alg in algs:
        for dataset in datasets:
            for net in nets:
                for seed in seeds:
                    # changes for each dataset
                    if dataset == "tiny_imagenet":
                        num_classes = 200
                        num_labels = label_amount[1] * num_classes
                        img_size = 32
                        crop_ratio = 0.875

                        lr = 5e-4
                        layer_decay = 0.5
                    else:
                        num_classes = 100
                        num_labels = label_amount[0] * num_classes
                        img_size = 32
                        crop_ratio = 0.875

                        lr = 5e-4
                        layer_decay = 0.5

                    if alg == "dtkd":
                        lr = 5e-3
                        
                    
                    port = dist_port[count]
                    cfg = create_ossl_cv_config(
                        alg,
                        seed,
                        dataset,
                        net,
                        num_classes,
                        num_labels,
                        img_size,
                        crop_ratio,
                        port,
                        lr,
                        weight_decay,
                        layer_decay,
                        warmup,
                        amp,
                    )
                    count += 1
                    create_configuration(cfg, config_file)

if __name__ == "__main__":
    if not os.path.exists("./saved_models/OSKD_cv/"):
        os.makedirs("./saved_models/OSKD_cv/", exist_ok=True)
    if not os.path.exists("./config/OSKD_cv/"):
        os.makedirs("./config/OSKD_cv/", exist_ok=True)
    label_amount = {"l":[100,100], "full":[500, 500]}
    for i in label_amount:
        exp_OSKD_cv(label_amount=label_amount[i])


