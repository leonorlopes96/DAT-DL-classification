

import os


import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight

import torch as th
from torch import nn
import torchinfo as ti
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.data.dat_pet_dataset import PETDataset
from src.data.dat_spect_dataset import SPECTDataset
from src.models.resnet import ResNet
from src.data.process_image_functions import create_slice_figure
from src.training.evaluate import get_predictions, performance_evaluate, calculate_ROC_curves


def train(weights_dir, labels, args):

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    classes = list(labels.values())

    df = pd.read_csv(args.train_data_dir)
    df = df[df['dataset'] == 'train']

    df = df[df['labels'] != 'PSP']
    df.insert(len(df.columns), 'gen_orig', 'gen')

    # TODO: try cross validation as well
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['labels'])

    train_data = SPECTDataset(train_df, labels)  # alter to SPECTDataset if train_gen_spect data is SPECT or genSPECT
    val_data = SPECTDataset(val_df, labels)

    loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)  # , num_workers=3)
    loader_val = DataLoader(val_data, batch_size=1, shuffle=True)


    train_weights = class_weight.compute_class_weight(class_weight='balanced', classes=list(labels.keys()),
                                                      y=train_df['labels'].values)
    print(train_weights)



    for img, label in loader_val:
        # print(img.shape)
        # img = img.float()[None, :, :, :, :]
        # img = th.permute(img, (1, 0, 2, 3, 4))
        create_slice_figure(img[0 ,: ,: ,:], str(label))
        break

    # Get Model
    model = ResNet(num_classes=len(classes))
    model = nn.DataParallel(model).to(device)
    ti.summary(model, input_size=(1, 1, 91, 109, 91))


    # Define the optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_drop,
                                                        patience=args.lr_patience, verbose=True)


    # Create the tensorboard summary writer.
    summary = SummaryWriter(args.train_dir + '/logs/', purge_step=0)

    # Iteration counter
    it = 0

    best_accuracy = 0
    best_loss = 100


    # Repeat training the given number of epochs
    for epoch in range(args.epochs):
        print(f"Starting epoch {epoch + 1}/{args.epochs}...")

        # Run one epoch
        for img, label in tqdm(loader_train):
            it += 1
            # loss=None
            # Run a training step

            img = img.float()[None, :, :, :, :]
            img = th.permute(img, (1, 0, 2, 3, 4))
            img = img.to(device)
            label = label.to(device)
            train_weights = th.Tensor(train_weights).to(device)

            # label = th.zeros(config.NUM_CLASSES, dtype=th.float).scatter_(dim=0, index=label.clone().detach(), value=1)
            # label = label.to(device)

            model.train()

            pred = model(img)

            # print(label)
            # print(train_weights)

            loss_function = nn.NLLLoss(reduction='sum', weight=train_weights)

            loss = loss_function(pred, label)

            # loss = categorical_crossentropy(pred, label)#, weight=train_weights)

            # print(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log the training loss once every 50 iterations
            if (it % 50) == 0:
                summary.add_scalar("loss", loss.detach().item(), it)

        y_true, y_pred, y_score, losses = get_predictions(model, loader_val, device)

        val_loss = np.mean(losses)
        val_accuracy = accuracy_score(y_true, y_pred)
        summary.add_scalar("accuracy", val_accuracy.item(), epoch)

        print(f'Loss: {val_loss}  |  Accuracy: {val_accuracy}')


        scheduler.step(val_accuracy)

        # save model state each 10 epoch
        if (epoch % 10) == 0:
            th.save(model.state_dict(), os.path.join(args.train_dir + '/logs/', f"model_{epoch}.pt"))

        # if val_accuracy > best_accuracy:
        if val_loss < best_loss:
            # Update patience and best_accuracy
            # best_accuracy = val_accuracy
            best_loss = val_loss
            patience = args.early_stop_patience

            th.save(model.state_dict(), weights_dir)

        else:
            patience -= 1

        print(f"My remaining patience is {patience}.")

        if patience == 0:
            print("My patience run out.")
            # y_true, y_pred, y_score = get_prediction(model, loader_val, device)
            # report = classification_report(y_true, y_pred, target_names=list(config.LABELS.keys()))
            # print(report)
            # perDict = performance_evaluate(y_true, y_pred)
            break


    print('Evaluation on validation data:')
    y_true, y_pred, y_score, losses = get_prediction(model, loader_val, device)
    report = classification_report(y_true, y_pred, target_names=list(labels.keys()))
    print(report)
    perDict = performance_evaluate(y_true, y_pred)

    roc_auc = calculate_ROC_curves(y_true, y_pred, y_score, labels_dict= labels, classes=list(labels.values()),
                                   save_dir=args.train_dir)

    with open(os.path.join(args.train_dir, 'metrics.txt'), mode='w') as file_handle:
        for key, value in labels.items():
            file_handle.write('ROCAUC_{}:{}'.format(key, roc_auc[value]))

        file_handle.write(report)

        file_handle.write('Sensitivity/recall: ' + str(perDict['TPR']) + '\n')
        file_handle.write('Specificity: ' + str(perDict['TNR']) + '\n')
        file_handle.write('Precision/positive predictive value: ' + str(perDict['PPV']) + '\n')
        file_handle.write('Negative predictive value: ' + str(perDict['NPV']) + '\n')