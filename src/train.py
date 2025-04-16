import os
import torch
import time
import numpy as np
from tqdm import tqdm
from opts import get_args
from core.dataset import create_Dataset, create_DataLoader
from sklearn.model_selection import StratifiedKFold
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, ConfigLogging, save_print_results, calculate_u_test, plot_tsne
from models.OverallModal import build_model
from core.metric import MetricsTop


opt = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


def train(opt, model, dataset, optimizer, loss_fn, epoch, metrics):
    dataLoader = create_DataLoader(opt, dataset)
    train_pbar = tqdm(dataLoader)

    losses = AverageMeter()
    y_pred, y_true = [], []
    model.train()

    for data in train_pbar:
        inputs = {
            'au': data['au'].to(device),
            'em': data['em'].to(device),
            'hp': data['hp'].to(device),
            'bp': data['bp'].to(device),
            'padding_mask': {
                'au': data['padding_mask_au'].to(device),
                'em': data['padding_mask_em'].to(device),
                'hp': data['padding_mask_hp'].to(device),
                'bp': data['padding_mask_bp'].to(device)
            },
            'length': {
                'au': data['au_lengths'].to(device),
                'em': data['em_lengths'].to(device),
                'hp': data['hp_lengths'].to(device),
                'bp': data['bp_lengths'].to(device)
            }
        }
        label = data['label'].to(device)    # quality ra readiness
        qu, ra, re = [], [], []
        for i in range(len(label)):
            qu.append(label[i][0])
            ra.append(label[i][1])
            re.append(label[i][2])
        label = {'quality': torch.tensor(qu).to(device), 'ra': torch.tensor(ra).to(device), 'readiness': torch.tensor(re).to(device)}
        batchsize = inputs['au'].shape[0]

        output, _, _ = model(inputs)
        # features = torch.mean(inputs['bp'], dim=1)
        # label = label[opt.labelType]
        # model.fit(features, label)

        loss_qu = loss_fn['quality'](output['quality'], label['quality'])
        loss_ra = loss_fn['ra'](output['ra'], label['ra'])
        loss_re = loss_fn['readiness'](output['readiness'], label['readiness'])
        loss = 0.2 * loss_qu + 1 * loss_ra + 0.2 * loss_re
        # loss = loss_fn[opt.labelType](output, label[opt.labelType])     # iTransformer loss computation
        losses.update(loss.item(), batchsize)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output[opt.labelType].cpu())
        # y_pred.append(output.cpu())
        y_true.append(label[opt.labelType].cpu())

        train_pbar.set_description('train')
        train_pbar.set_postfix({
            'epoch': '{}'.format(epoch),
            'loss': '{:.5f}'.format(losses.value_avg),
            'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
        })

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)

    return train_results


def test(opt, model, dataset, optimizer, loss_fn, epoch, metrics):
    dataLoader = create_DataLoader(opt, dataset)
    test_pbar = tqdm(dataLoader)

    losses = AverageMeter()
    y_pred, y_true = [], []
    model.eval()

    with torch.no_grad():
        input_tsne, input_fusion_tsne = [], []
        for data in test_pbar:
            inputs = {
                'au': data['au'].to(device),
                'em': data['em'].to(device),
                'hp': data['hp'].to(device),
                'bp': data['bp'].to(device),
                'padding_mask': {
                    'au': data['padding_mask_au'].to(device),
                    'em': data['padding_mask_em'].to(device),
                    'hp': data['padding_mask_hp'].to(device),
                    'bp': data['padding_mask_bp'].to(device)
                },
                'length': {
                    'au': data['au_lengths'].to(device),
                    'em': data['em_lengths'].to(device),
                    'hp': data['hp_lengths'].to(device),
                    'bp': data['bp_lengths'].to(device)
                }
            }
            label = data['label'].to(device)
            batchsize = inputs['au'].shape[0]

            output, data_tsne, data_fusion_tsne = model(inputs)
            input_tsne.append(data_tsne)
            input_fusion_tsne.append(data_fusion_tsne)
            # features = torch.mean(inputs['bp'], dim=1)
            # output = model.predict(features)

            loss = loss_fn[opt.labelType](output[opt.labelType], label)
            # loss = loss_fn[opt.labelType](output, label)
            losses.update(loss.item(), batchsize)

            y_pred.append(output[opt.labelType].cpu())
            # y_pred.append(output.cpu())
            # y_pred.append(torch.tensor(output).cpu())
            y_true.append(label.cpu())

            test_pbar.set_description('test')
            test_pbar.set_postfix({
                'epoch': '{}'.format(epoch),
                'loss': '{:.5f}'.format(losses.value_avg),
                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            })

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        # if epoch == 11:
        #     calculate_u_test(pred, true)
        test_results = metrics(pred, true)
        # plot_tsne(data=input_tsne, fusion=input_fusion_tsne, path='')

    return test_results


def main(parse_args):
    opt = parse_args

    log_path = os.path.join(opt.log_path, opt.datasetName.upper())
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, time.strftime('%Y-%m-%d-%H:%M:%S' + '.log'))
    logger = ConfigLogging(log_file)
    logger.info(opt)    # 保存当前模型参数

    setup_seed(opt.seed)
    dataset = create_Dataset(opt.datasetName, opt.labelType).d_l

    Avg_Accuracy = []
    Avg_F1_score = []
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(dataset['data'], dataset['label']):
        X_train = np.array(dataset['data'])[train_index]
        Y_train = np.array(dataset['multi_task'])[train_index]
        # Y_train = np.array(dataset['label'])[train_index]

        X_test = np.array(dataset['data'])[test_index]
        # Multi_test = np.array(dataset['multi_task'])[test_index]
        Y_test = np.array(dataset['label'])[test_index]

        model = build_model(opt).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt.learning_rate,
            weight_decay=opt.weight_decay
        )

        loss_fn = {'quality': torch.nn.CrossEntropyLoss(), 'ra': torch.nn.CrossEntropyLoss(), 'readiness': torch.nn.CrossEntropyLoss()}
        metrics = MetricsTop().getMetics(opt.datasetName)
        scheduler_warmup = get_scheduler(optimizer, opt.epochs)

        for epoch in range(1, opt.epochs+1):
            train_results = train(opt, model, [X_train, Y_train], optimizer, loss_fn, epoch, metrics)
            test_results = test(opt, model, [X_test, Y_test], optimizer, loss_fn, epoch, metrics)
            save_print_results(opt, logger, train_results, test_results)
            scheduler_warmup.step()

        Avg_Accuracy.append(test_results['Accuracy'])
        Avg_F1_score.append(test_results['F1-Score'])

    logger.info(f"mean acc: {np.array(Avg_Accuracy).mean()}")
    logger.info(f"mean f1 : {np.array(Avg_F1_score).mean()}")


if __name__ == '__main__':
    main(opt)
