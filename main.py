from dataset_loader import get_cifar10_dataloaders, get_dataloader
from dataset_loader import get_cifar100_dataloaders
from dataset_loader import get_imagenet_dataloaders, get_fgnet_dataloaders, get_hic_dataloaders

from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
from time import time
import torch
import losses, dataset_loader,data
from models import MLP
from model import *
from metrics import *
import argparse
from snn_models import SNNResNet18,mtsnnMLP


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--datadir', default="data/")
parser.add_argument('--loss', required=True)
parser.add_argument('--REP', type=int, required=True)
parser.add_argument('--output', default='output/model.pth')
parser.add_argument('--lamda', type=float)
parser.add_argument('--batchsize', type=int, default=32)
#parser.add_argument('--classes_num', type=int, default=70)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--model', default="resnet18_weights", type=str)
parser.add_argument('--use_pre_net', action='store_true', 
                    help="Use pretrained network if set") #python train.py --use_pre_net
parser.add_argument('--learnable_params', type=bool, default=False)
parser.add_argument('--use_original_loss', type=bool, default=False)

args = parser.parse_args()



device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################## DATASET ##############################

train_transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((int(256*1.05), int(256*1.05)), antialias=True),
    transforms.RandomCrop((256, 256)),
    transforms.ColorJitter(0.1, 0.1),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((256, 256), antialias=True),  # Direct resize, no overcrop
    # Or use CenterCrop if you want to match training:
    # transforms.Resize((int(256*1.05), int(256*1.05)), antialias=True),
    # transforms.CenterCrop((256, 256)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])



drs = getattr(data, args.dataset)
ds = drs(args.datadir, train_transform, 'train', args.REP)
K = ds.K
num_classes = K
train_loader = DataLoader(ds, args.batchsize, True, num_workers=8, pin_memory=True)

ts = drs(args.datadir, test_transform, 'test', args.REP)
test_loader = DataLoader(ts, 64, num_workers=4, pin_memory=True)


###############################################################################




if args.lamda is None:
    loss_fn = getattr(losses, args.loss)(num_classes)
else:
    try:
        loss_fn = getattr(losses, args.loss)(num_classes, args.lamda)
    except:
        loss_fn = getattr(losses, args.loss)(num_classes)



from omegaconf import OmegaConf 
cfg = OmegaConf.create({
    'loss': args.loss,  #  
    'model': {
        'neuron': {
            'beta': 0.9
        },
        'time_steps': 4,
        'layer': {
            'dropout_rate': 0.5
        }
    }
})




out_features = loss_fn.how_many_outputs()  # learnable_params

#################### model ##########################


model = SNNResNet18(cfg, num_classes=K)

if ds.modality == 'tabular':
    #model = MLP(ds[0][0].shape[0], 128, loss_fn.how_many_outputs())
    #print ('loss_fn.how_many_outputs()',loss_fn.how_many_outputs())
    model = mtsnnMLP(ds[0][0].shape[0], 128, loss_fn.how_many_outputs())
    epochs = 1000 #1000
else:
    model.fc = torch.nn.Linear(512, loss_fn.how_many_outputs())
    epochs = 200 #100
loss_fn.to(device)
model = model.to(device)
model.loss_fn = loss_fn







############################################




loss_fn.to(device)
model = model.to(device)
model.loss_fn = loss_fn

opt = torch.optim.Adam(model.parameters(), args.lr)


train_time = 0
test_time = 0
Best_test_acc = []

#early stop
best_test_loss = float('inf')
patience_counter = 0
patience = 20
All_metrics = [] 
for epoch in range(args.epochs):
    model.train()
    if epoch % 50 == 0:
        print(f'* Epoch {epoch+1}/{args.epochs}')
    tic = time()
    avg_loss = 0
    correct = 0
    total = 0
    for X, Y in train_loader:
        X = X.to(device)
        Y = Y.to(device)
        Yhat = model(X)
        _, predicted = torch.max(Yhat, 1)
        total += Y.size(0)
        correct += (predicted == Y).sum().item()
        loss_value = loss_fn(Yhat, Y).mean()
        opt.zero_grad()
        loss_value.backward()
        opt.step()
        avg_loss += float(loss_value) / len(train_loader)

    toc = time()
    accuracy = 100 * correct / total
    if epoch % 50 == 0:
        print(f'- {toc-tic:.0f}s - train_Loss: {avg_loss:.4f} - train_Acc: {accuracy:.2f}%')
    train_time += toc-tic
#################################
    model.eval() 
    test_tic = time()
    test_loss = 0
    test_correct = 0
    test_total = 0
    ppreds, ypreds, ytrues = [], [], []
    YY_pred = []
    YY_true = []
    with torch.no_grad(): 
        for X, Y in test_loader:
            X = X.to(device)
            Y = Y.to(device)
           
            Yhat = model(X)
            YY_pred.append(Yhat)
            YY_true.append(Y)
            #print ('Yhat.shape',Yhat.shape,Yhat[0],Y)
            loss_value = loss_fn(Yhat, Y).mean()

            test_loss += float(loss_value) / len(test_loader)
            _, predicted = torch.max(Yhat, 1)
            probs = F.softmax(Yhat, dim=1)
            p = probs.detach().cpu()
            #ypreds.append(predicted.cpu())
            #ytrues.append(Y.cpu())
            #ppreds.append(p.cpu())

            test_total += Y.size(0)
            #test_correct += (predicted == Y).sum().item()
        YY_pred = torch.cat(YY_pred)
        PP_pred = loss_fn.to_proba(YY_pred)
        YY_pred = loss_fn.to_classes(YY_pred)
        YY_true = torch.cat(YY_true)
    test_toc = time()
    #test_accuracy = 100 * test_correct / test_total
    #Best_test_acc.append(test_accuracy)
    test_time += test_toc - test_tic

    ppreds = PP_pred
    ypreds = YY_pred
    ytrues = YY_true
    #import numpy as np
    #print(f"ppreds shape: {ppreds.shape}, ypreds shape: {ypreds.shape}, ytrues shape: {ytrues.shape}")
    #print(f"Unique labels: {torch.unique(ytrues)}")
    #print(f"Number of classes in model: {ppreds.shape[-1]}")
    metrics = {
        "epoch": epoch + 1,
        "Acc": float(acc(ppreds, ypreds, ytrues)),
        "MAE": float(mae(ppreds, ypreds, ytrues)),
        "F1": float(f1(ppreds, ypreds, ytrues)),
        "QWK": float(quadratic_weighted_kappa(ppreds, ypreds, ytrues)),
        "ZME": float(zero_mean_error(ppreds, ypreds, ytrues)),
        "NLL": float(negative_log_likelihood(ppreds, ypreds, ytrues)),
        "Unimodal": float(times_unimodal_wasserstein(ppreds, ypreds, ytrues)),
        "kendall_tau": float(kendall_tau(ppreds, ypreds, ytrues))
    }
    #except:
    #    print("Some metrics calculation failed.",K)
    #    break

    All_metrics.append(metrics)

    if epoch % 50 == 0:
        #print(f'- {test_toc-test_tic:.0f}s - test_Loss: {test_loss:.4f} - test_Acc: {test_accuracy:.2f}%')
        print(f'- {test_toc-test_tic:.0f}s - test_Loss: {test_loss:.4f}')

    #early stopping check
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

time = test_time + train_time
#model.train_time = train_time
#torch.save(model.cpu(), args.output)






#Best_ta = max(Best_test_acc)
best_metrics = max(All_metrics, key=lambda x: x["Acc"])
print(f"best_metrics in {best_metrics['epoch']} epoch: {best_metrics}")

print(f'dataset: {args.dataset}, loss_fn: {args.loss}, non_pre_weighted\n'
      f'batch_size: {args.batchsize}, lr: {args.lr}, epochs: {args.epochs}\n'
      f'REP: {args.REP}\n'
      f'time: {time:.2f}s\n'
      f'use_original_loss: {args.use_original_loss}\n'
      f'batch_size: {args.batchsize}\n'
      f'lamda: {args.lamda if args.lamda is not None else "None"}\n')
'''
print(f'dataset: {args.dataset}, loss_fn: {args.loss}, non_pre_weighted\n'
      f'batch_size: {args.batchsize}, lr: {args.lr}, epochs: {args.epochs}\n'
      f'time: {time:.2f}s, model: {args.model}\n'
      f'lamda: {args.lamda if args.lamda is not None else "None"},use_original_loss: {args.use_original_loss}\n'
      f'fc_layer_name: {args.fc_layer_name}, use_leaky_gaussianfc:{args.use_leaky_gaussianfc}\n'
      f'batch_size: {args.batchsize},Best_test_acc: {Best_ta}, Best_test_acc_index: {Best_test_acc.index(Best_ta)}\n')
'''

#torch.save(model.state_dict(), args.output)

