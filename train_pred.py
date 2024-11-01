from __future__ import print_function, division
import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()

device = torch.device('cuda')


def train_model(train_data, val_data, config, model, criterion, optimizer, scheduler, n_fold, num_epochs):

    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        path = os.path.join(config.path,  'model_na_np_' + str(n_fold) + '_' + str(epoch) + '.pt')

        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode

        training_loss = 0.0
        training_corrects = 0
        val_corrects = 0
        total_train = 0
        total_val = 0
        all_val_labels = torch.FloatTensor()
        all_val_labels = all_val_labels.to(device)
        all_test_labels = torch.FloatTensor()
        all_val_pred = torch.FloatTensor()
        all_val_pred = all_val_pred.to(device)

        # Iterate over data.
        for i, (inputs, labels) in enumerate(train_data,0):
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.squeeze_()
            loss_train = criterion(outputs, labels)
            loss_train.backward()
            optimizer.step()

            # Metrics: loss and acc
            _, preds = torch.max(outputs, 1)
            training_loss += loss_train.item() * inputs.size(0)
            training_corrects += torch.sum(preds == labels.data)
            total_train += labels.size(0)
        scheduler.step()

        train_loss = training_loss / total_train
        train_acc = training_corrects.double() / total_train

        torch.save(model.state_dict(), path)

        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_data, 0):
                inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)
                bs, c, h, w = inputs.size()
                labels = labels.squeeze_()
                all_val_labels= torch.cat((all_val_labels,labels),0)
                # deep copy the model
                outputs = model(inputs)
                varOutput= outputs.view(bs, -1)
                val_loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                all_val_pred=torch.cat((all_val_pred,preds),0)
                val_corrects += torch.sum(preds == labels.data)
                total_val += labels.size(0)


            val_acc = val_corrects.double() / total_val
            time_elapsed = time.time() - since

            if val_acc > best_acc:
                best_acc = val_acc
            print('Epoch:{:d} Train Acc.:{:.4f}, Train Loss.:{:.4f},  Val Acc.:{:.4f}, Val Loss.:{:.4f}, Training time.:{:.4f}'.format(epoch, train_acc, train_loss, val_acc, val_loss, time_elapsed))


            #best_model_wts = copy.deepcopy(model.state_dict())


            writer.add_scalar("Train Loss", train_loss, epoch)
            writer.add_scalar("Train Accuracy", train_acc, epoch)
            writer.add_scalar("Loss", val_loss, epoch)
            writer.add_scalar("Accuracy", val_acc, epoch)

    writer.close()

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #model.load_state_dict(best_model_wts)

    return all_val_pred, all_val_labels, all_val_pred, all_val_labels

def pred_val(test_data, model):
    all_pred = torch.FloatTensor()
    all_pred = all_pred.to(device)
    all_img = torch.FloatTensor()

    with torch.no_grad():
        for i, (inputs) in enumerate(test_data, 0):
            inputs= inputs.to(device, dtype=torch.float)
            bs, c, h, w = inputs.size()
            outputs = model(inputs)
            varOutput = outputs.view(bs, -1)
            _, preds = torch.max(outputs, 1)
            all_pred = torch.cat((all_pred, preds), 0)

    return all_pred