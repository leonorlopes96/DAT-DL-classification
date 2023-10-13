
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report

import torch as th
from torch import nn
from torch.utils.data import DataLoader

from src.data.dat_spect_dataset import SPECTDataset
from src.data.process_image_functions import create_slice_figure
from src.models.resnet import ResNet

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"


def test(labels, load_weights_dir, args):

    classes = list(labels.values())
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Data
    test_df = pd.read_csv(args.test_data_dir)
    test_df.insert(len(test_df.columns), 'gen_orig', 'origSPECT')

    test_data = SPECTDataset(test_df, labels)

    loader_test = DataLoader(test_data, batch_size=1, shuffle=True)  # , num_workers=3)

    for img, label in loader_test:
        create_slice_figure(img[0, :, :, :], str(label))
        break

    # Get Model
    model = ResNet(num_classes=len(classes))
    model = nn.DataParallel(model).to(device)

    state_dict_cc = th.load(load_weights_dir)
    model.load_state_dict(state_dict_cc)

    y_true, y_pred, y_score, losses = get_predictions(model, loader_test, device)

    np.save(os.path.join(args.test_dir, 'y_pred.npy'), y_pred)
    np.save(os.path.join(args.test_dir, 'y_true.npy'), y_true)
    np.save(os.path.join(args.test_dir, 'y_score.npy'), y_score)

    report = classification_report(y_true, y_pred, target_names=list(labels.keys()))

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=['NC', 'PD'], cmap=plt.cm.Blues)
    plt.savefig(os.path.join(args.test_dir, 'confusion_matrix.png'))

    perDict = performance_evaluate(y_true, y_pred)

    roc_auc = calculate_ROC_curves(y_true, y_pred, y_score, classes=list(labels.values()), save_dir=args.test_dir)

    with open(os.path.join(args.test_dir, 'metrics.txt'), mode='w') as file_handle:
        for key, value in labels.items():
            file_handle.write('ROCAUC_{}:{}'.format(key, roc_auc[value]))

        file_handle.write(report)

        file_handle.write('Sensitivity/recall: ' + str(perDict['TPR']) + '\n')
        file_handle.write('Specificity: ' + str(perDict['TNR']) + '\n')
        file_handle.write('Precision/positive predictive value: ' + str(perDict['PPV']) + '\n')
        file_handle.write('Negative predictive value: ' + str(perDict['NPV']) + '\n')



def get_predictions(model: nn.Module, loader: DataLoader, device:th.device) -> (th.Tensor, th.Tensor, th.Tensor):

    with th.no_grad():
        y_true_ = []
        y_score_ = []
        y_pred_ = []
        losses = []

        for img, label in tqdm(loader):
            img = img.float()[None, :, :, :, :]
            img = th.permute(img, (1, 0, 2, 3, 4))
            img = img.to(device)
            label = label.to(device)


            model.eval()
            score = model(img)
            loss_function = nn.NLLLoss(reduction='sum')
            loss = loss_function(score, label)

            pred = th.argmax(score, dim=1)


            pred = pred.cpu()
            label = label.cpu()
            score = score.cpu()
            #loss = loss.cpu()

            y_true_.append(label)
            y_pred_.append(pred)
            y_score_.append(score)
            losses.append(loss.item())


        y_pred = th.cat(y_pred_, dim=0)
        y_true = th.cat(y_true_, dim=0)
        y_score = th.cat(y_score_, dim=0)
        #y_loss = th.cat(losses_, dim=0)


        # print(f'y_true {y_true.shape}')
        # print(f'y_pred {y_pred.shape}')
        # print(f'y_score {y_score.shape}')

    return y_true, y_pred, y_score, losses

def calculate_ROC_curves(y_true:th.Tensor, y_score:th.Tensor = None, labels_dict:dict = None, classes:list = None, save_dir:str = None) -> dict:
    """
    Calculate ROC curves and AUC per class and plot them
    :param y_true: ground truth labels
    :param y_pred: predictions
    :param classes: list of classes like [0, 1, 2]
    :param save_dir: directory to save the ROC curve plots
    :return: dict ROC AUC per class
    """
    # Transform labels to categorical
    #print(y_true)
    #y_true = label_binarize(y_true, classes=classes)
    #y_true = tf.keras.utils.to_categorical(y_true, num_classes=config.NUM_CLASSES)
    y_true = th.nn.functional.one_hot(y_true)

    # Calculate fpr and tpr for each class and save in dict
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    for cls in classes:
        plt.figure()

        plt.plot(
            fpr[cls],
            tpr[cls],
            color="darkorange",
            label="ROC curve (area = %0.2f)" % roc_auc[cls])

        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title('Receiver operating characteristic: {}'.format(list(labels_dict.keys())[cls]))
        plt.legend(loc="lower right")
        plt.savefig(save_dir + f'/ROC_{list(labels_dict.keys())[cls]}')
        #plt.show()
        plt.close()

    return roc_auc


def performance_evaluate(y_true:th.Tensor, y_pred:th.Tensor) -> dict:
    """
    Calculate the TP, FP, TN, FN from ground truth (y_orig) and prediction (Y_predict)
    :param y_true: ground truth labels
    :param y_pred: predictions
    :return: dict with several metrics
    """
    eplison = 1e-6 # To avoid division by zero
    perDict = dict()
    conf_matrix = confusion_matrix(y_true, y_pred)
    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    TP = np.diag(conf_matrix)
    TN = conf_matrix.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN + eplison)
    # Specificity or true negative rate
    TNR = TN / (TN + FP + eplison)
    # Precision or positive predictive value
    PPV = TP / (TP + FP + eplison)
    # Negative predictive value
    NPV = TN / (TN + FN + eplison)
    # Fall out or false positive rate
    FPR = FP / (FP + TN + eplison)
    # False negative rate
    FNR = FN / (TP + FN + eplison)
    # False discovery rate
    FDR = FP / (TP + FP + eplison)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN + eplison)

    perDict['FP'] = FP
    perDict['FN'] = FN
    perDict['TP'] = TP
    perDict['TN'] = TN
    perDict['TPR'] = TPR
    perDict['TNR'] = TNR
    perDict['PPV'] = PPV
    perDict['NPV'] = NPV
    perDict['FPR'] = FPR
    perDict['FNR'] = FNR
    perDict['FDR'] = FDR
    perDict['ACC'] = ACC

    print('Sensitivity/recall:' + str(TPR))
    print('Specificity:' + str(TNR))
    print('Precision/positive predictive value:' + str(PPV))
    print('Negative predictive value:' + str(NPV))



    return perDict
