import os
import sys
import datetime
import pathlib
import math
import numpy as np
import imageio

from PIL import Image
import scipy.stats as stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, cohen_kappa_score, jaccard_score

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from config import *
from utils import *

from dataloader import NGTrain, NGTest, NGValid
from networks.factory import model_factory

Image.MAX_IMAGE_PIXELS = None



def test_full_map(test_loader, net, epoch, output_path):
    # Setting network for evaluation mode.
    net.eval()

    prob_im = np.zeros([test_loader.dataset.labels.shape[0],
                        test_loader.dataset.labels.shape[1],
                        test_loader.dataset.labels.shape[2], test_loader.dataset.num_classes], dtype=np.float32)
    occur_im = np.zeros([test_loader.dataset.labels.shape[0],
                         test_loader.dataset.labels.shape[1],
                         test_loader.dataset.labels.shape[2], test_loader.dataset.num_classes], dtype=int)

    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(test_loader):
            # Obtaining images, labels and paths for batch.
            inps, labs, cur_maps, cur_xs, cur_ys = data

            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            outs = net(inps_c)
            soft_outs = F.softmax(outs, dim=1)

            for j in range(outs.shape[0]):
                cur_map = cur_maps[j]
                cur_x = cur_xs[j]
                cur_y = cur_ys[j]

                soft_outs_p = soft_outs.permute(0, 2, 3, 1).cpu().detach().numpy()

                prob_im[cur_map][cur_x:cur_x + test_loader.dataset.crop_size,
                                 cur_y:cur_y + test_loader.dataset.crop_size, :] += soft_outs_p[j, :, :, :]
                occur_im[cur_map][cur_x:cur_x + test_loader.dataset.crop_size,
                                  cur_y:cur_y + test_loader.dataset.crop_size, :] += 1

        # normalize to remove non-predicted pixels - if there is one
        occur_im[np.where(occur_im == 0)] = 1

        # calculate predictions
        prob_im_argmax = np.argmax(prob_im / occur_im.astype(float), axis=-1)
        # pixels with classes not used in the prediction are converted into 0
        prob_im_argmax[np.where(test_loader.dataset.labels == 2)] = 0

        for k, img_n in enumerate(test_loader.dataset.images):
            # Saving predictions.
            imageio.imwrite(os.path.join(output_path, img_n + '_pred_epoch_' + str(epoch) + '.png'),
                            prob_im_argmax[k]*255)

        lbl = test_loader.dataset.labels.flatten()
        pred = prob_im_argmax.flatten()
        print(lbl.shape, np.bincount(lbl.flatten()), pred.shape, np.bincount(pred.flatten()))

        acc = accuracy_score(lbl, pred)
        conf_m = confusion_matrix(lbl, pred)
        f1_s_w = f1_score(lbl, pred, average='weighted')
        f1_s_micro = f1_score(lbl, pred, average='micro')
        f1_s_macro = f1_score(lbl, pred, average='macro')
        kappa = cohen_kappa_score(lbl, pred)
        jaccard = jaccard_score(lbl, pred)
        tau, p = stats.kendalltau(lbl, pred)

        _sum = 0.0
        for k in range(len(conf_m)):
            _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)
        nacc = _sum / float(test_loader.dataset.num_classes)

        print("---- Validation/Test -- Epoch " + str(epoch) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Overall Accuracy= " + "{:.4f}".format(acc) +
              " Normalized Accuracy= " + "{:.4f}".format(nacc) +
              " F1 score weighted= " + "{:.4f}".format(f1_s_w) +
              " F1 score micro= " + "{:.4f}".format(f1_s_micro) +
              " F1 score macro= " + "{:.4f}".format(f1_s_macro) +
              " Kappa= " + "{:.4f}".format(kappa) +
              " Jaccard= " + "{:.4f}".format(jaccard) +
              " Tau= " + "{:.4f}".format(tau) +
              " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
              )

        sys.stdout.flush()

    return acc, nacc, f1_s_w, kappa, conf_m



def validate(validation_loader, net, epoch, output_path):
    # Setting network for evaluation mode.
    net.eval()
    
    # pad for image
    prob_im = np.zeros([validation_loader.dataset.org_mask.shape[0],
                        validation_loader.dataset.org_mask.shape[1],
                        validation_loader.dataset.org_mask.shape[2], validation_loader.dataset.num_classes], dtype=np.float32)
    
    prob_im = np.rot90(prob_im)
    #print("prob_im shape", prob_im.shape) # (1, 4000, 4000, 2)
    #print("prob_im 0s", prob_im) # (1, 4000, 4000, 2)
    #print("unique vals",np.unique(prob_im))
    
    occur_im = np.zeros([validation_loader.dataset.org_mask.shape[0],
                         validation_loader.dataset.org_mask.shape[1],
                         validation_loader.dataset.org_mask.shape[2], validation_loader.dataset.num_classes], dtype=int)
    
    occur_im = np.rot90(occur_im)
    # pad for mask
    #labs_pad = np.zeros([validation_loader.dataset.org_mask.shape[0],
    #                    validation_loader.dataset.org_mask.shape[1],
    #                    validation_loader.dataset.org_mask.shape[2], validation_loader.dataset.num_classes], dtype=np.float32)
    

    #labs_occur_pad = np.zeros([validation_loader.dataset.org_mask.shape[0],
    #                     validation_loader.dataset.org_mask.shape[1],
    #                     validation_loader.dataset.org_mask.shape[2], validation_loader.dataset.num_classes], dtype=int)
    


    with torch.no_grad():
        # Iterating over batches.
        for i, data in enumerate(validation_loader):
            print("what is i", i)
            print("what is data len", len(data)) # is list
            
            # Obtaining images, labels and paths for batch.
            inps, labs, cur_maps, xmins, ymaxs, xmaxs, ymins = data    

            #print("What is i?", i) # 0
            #print("data length", len(data)) # 5
            #print("inps shape", inps.shape) # torch.Size([1, 23, 250, 250])
            
            # Casting to cuda variables.
            inps_c = Variable(inps).cuda()
            # labs_c = Variable(labs).cuda()

            # Forwarding.
            outs = net(inps_c) # outs = 4 dims (0 = idx ,1 = c, 2 = x, 3 = y) 
            # IS OUTS an array? an iterable? or patches? 
            # THEY ARE PATCHES
            #print("outs shape", outs.shape) # torch.Size([1, 2, 250, 250])
            #print("outs[0].shape?",outs[0].shape) #  torch.Size([2, 250, 250])
            #print("outs[1].shape?",outs[1].shape) # only outs[0] has shape now 
            #print("outs[2].shape?", outs.shape[2])

            soft_outs = F.softmax(outs, dim=1) # softmax calcs on channels
            print("soft outs shape before forloop", soft_outs.shape)
            # IS OUTS an array? an iterable? or patches? 
            # THEY ARE PATCHES
            print("what is i before forloop j", i)
            #print("what is data shape before forloop j", data.shape) # is list
            
            print("cur_maps before forloop j", cur_maps)
            
            for j in range(outs.shape[0]):
                #print("what is j", j) #   is int
                #print("type of j", type(j))
                print(range(outs.shape[0])) #(0, 5)
                cur_map = cur_maps[j]
                xmin = xmins[j]
                ymax = ymaxs[j]
                xmax = xmaxs[j]
                ymin = ymins[j]
                print("cur_map",cur_map, "xmin:", xmin, "ymax", ymax, "xmax", xmax, "ymin", ymin)
                print("cur_maps after forloop j", cur_maps)
                print("cur_map after forloop j", cur_map)
                
                soft_outs_p = soft_outs.permute(0, 2, 3, 1).cpu().detach().numpy()
                print("soft_outs_p on forloop", soft_outs_p.shape) #(5, 250, 250, 2)
                #print("soft_outs_p shape", soft_outs_p.shape) # (1, 250, 250, 2)
                #print("soft_outs_p[0].shape?",soft_outs_p[0].shape) # (250, 250, 2)
                #print("soft_outs_p", soft_outs_p)
                #print("soft_outs_p unique vals", np.unique(soft_outs_p))

                #img = Image.fromarray(soft_outs_p[0], 'RGB')
                #img.save('my.png')
                #img.show()
                
                prob_im[:, xmin:xmax, ymax:ymin, :] += soft_outs_p[j, :, :, :]
                #print("prob_im recon shape", prob_im.shape)
                #print("prob_im after", prob_im)
                #printt("prob_im unique vals after", prob_im)

                occur_im[:, xmin:xmax, ymax:ymin, :] += 1
                #print("occur_im recon", occur_im.shape)

                #labs_pad[cur_map][xmin:xmax, ymax:ymin, :] += labs[j, :, : , :]
               
                #labs_occur_pad[cur_map][xmin:xmax, ymax:ymin, :] += 1

            # Wrong recreation but trains and validates
            #for j in range(outs.shape[0]):
            #    cur_map = cur_maps[j]
            #    cur_x = cur_xs[j]
            #    cur_y = cur_ys[j]

            #    soft_outs_p = soft_outs.permute(0, 2, 3, 1).cpu().detach().numpy()


            #    prob_im[:, cur_x:cur_x + validation_loader.dataset.crop_size, 
            #            cur_y:cur_y + validation_loader.dataset.crop_size, :] += soft_outs_p[j, :, :, :]
            #
            #    occur_im[:, cur_x:cur_x + validation_loader.dataset.crop_size, 
            #            cur_y:cur_y + validation_loader.dataset.crop_size, :] += 1
        

        print("what is i after forloop j", i)
        print("what is data after forloop j", data)
        # normalize to remove non-predicted pixels - if there is one
        occur_im[np.where(occur_im == 0)] = 1
        # occur_im are just the patches without calcu?

        # calculate predictions
        # np.argmax RETURNS THE INDICES OF THE MAX ELEMENT, so axis = -1 are the channels
        prob_im_argmax = np.argmax(prob_im / occur_im.astype(float), axis=-1) # matrix division element wise
        # = output is prds on train (line 267 in main.py) 
        # HERE IT IS ACTUALLY/ SHOULD BE THE WHOLE IMAGE ARRAY
        
        # pixels with classes not used in the prediction are converted into 0
        prob_im_argmax[np.where(validation_loader.dataset.labels == 2)] = 0

        #THIS WILL TAKE dataset.images which is a whole image 4000x4000x23
        # SO WE NEED TO PUT TOGETHER 
        # WORK AFTER IMAGE ARRAY CREATION IS DONE: THIS IS IMAGE SAVING
        #for k, img_n in enumerate(validation_loader.dataset.images):
            # Saving predictions.
        imageio.imwrite(os.path.join(output_path, 'im' + '_pred_epoch_' + str(epoch) + '.png'),
                            prob_im_argmax[0]*255)
        # LINE 71  WILL BE THEN prob_im_argmax*255
            
        
        # lbl since originally we are passing 1 big image mask, this will contain all the values of 
        # the mask patches combined (side by side) 
        # as labels is an array of 256 patches of 250x250 (256, 250, 250)
        ###---lbl FLATTENING IS OK---###
        
        # labels (lbl) here are patchwise so pred should also be patchwise
        ###NUMBER82 ---UNLESS WE MAKE an array for labels (recreate the mask as well then flatten)---###
        lbl = validation_loader.dataset.org_mask.flatten() # = labels on train
        
        # so soft_outs_p should be flattened instead of pred
        ###---FLATTEN prob_im_argmax if NUMBER82 is possible (which makes more sense)---### 
        pred = prob_im_argmax.flatten()  # now flattened like prds on train
        print(lbl.shape, np.bincount(lbl.flatten()), pred.shape, np.bincount(pred.flatten()))

        acc = accuracy_score(lbl, pred)
        conf_m = confusion_matrix(lbl, pred)
        f1_s_w = f1_score(lbl, pred, average='weighted')
        f1_s_micro = f1_score(lbl, pred, average='micro')
        f1_s_macro = f1_score(lbl, pred, average='macro')
        kappa = cohen_kappa_score(lbl, pred)
        jaccard = jaccard_score(lbl, pred)
        tau, p = stats.kendalltau(lbl, pred)

        _sum = 0.0
        for k in range(len(conf_m)):
            _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)
        nacc = _sum / float(validation_loader.dataset.num_classes)

        print("---- Validation/Test -- Epoch " + str(epoch) +
              " -- Time " + str(datetime.datetime.now().time()) +
              " Overall Accuracy= " + "{:.4f}".format(acc) +
              " Normalized Accuracy= " + "{:.4f}".format(nacc) +
              " F1 score weighted= " + "{:.4f}".format(f1_s_w) +
              " F1 score micro= " + "{:.4f}".format(f1_s_micro) +
              " F1 score macro= " + "{:.4f}".format(f1_s_macro) +
              " Kappa= " + "{:.4f}".format(kappa) +
              " Jaccard= " + "{:.4f}".format(jaccard) +
              " Tau= " + "{:.4f}".format(tau) +
              " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
              )

        sys.stdout.flush()

    return acc, nacc, f1_s_w, kappa, conf_m




def train(train_loader, model, criterion, optimizer, epoch):
    # Setting network for training mode.
    model.train()

    # Average Meter for batch loss.
    train_loss = list()

    # Iterating over batches.
    for i, data in enumerate(train_loader):
        # Obtaining images, labels and paths for batch.
        inps, labels = data[0], data[1]

        # if the current batch does not have samples from all classes
        # print('out i', i, len(np.unique(labels.flatten())))
        # if len(np.unique(labels.flatten())) < 10:
        #     print('in i', i, len(np.unique(labels.flatten())))
        #     continue

        # Casting tensors to cuda.
        inps = Variable(inps).cuda()
        labs = Variable(labels).cuda()

        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        outs = model(inps)

        # Computing loss.
        loss = criterion(outs, labs)

        if math.isnan(loss):
            print('-------------------------NaN-----------------------------------------------')
            print(inps.shape, labels.shape, outs.shape, np.bincount(labels.flatten()))
            print(np.min(inps.cpu().data.numpy()), np.max(inps.cpu().data.numpy()),
                  np.isnan(inps.cpu().data.numpy()).any())
            print(np.min(labels.cpu().data.numpy()), np.max(labels.cpu().data.numpy()),
                  np.isnan(labels.cpu().data.numpy()).any())
            print(np.min(outs.cpu().data.numpy()), np.max(outs.cpu().data.numpy()),
                  np.isnan(outs.cpu().data.numpy()).any())
            print('-------------------------NaN-----------------------------------------------')
            raise AssertionError

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())

        # Printing.
        if (i + 1) % DISPLAY_STEP == 0:
            soft_outs = F.softmax(outs, dim=1)
            # Obtaining predictions.
            prds = soft_outs.cpu().data.numpy().argmax(axis=1).flatten()

            labels = labels.cpu().data.numpy().flatten()

            # filtering out pixels
            coord = np.where(labels != train_loader.dataset.num_classes)
            labels = labels[coord]
            prds = prds[coord]

            acc = accuracy_score(labels, prds)
            conf_m = confusion_matrix(labels, prds, labels=[0, 1])
            f1_s = f1_score(labels, prds, average='weighted')

            _sum = 0.0
            for k in range(len(conf_m)):
                _sum += (conf_m[k][k] / float(np.sum(conf_m[k])) if np.sum(conf_m[k]) != 0 else 0)

            print("Training -- Epoch " + str(epoch) + " -- Iter " + str(i + 1) + "/" + str(len(train_loader)) +
                  " -- Time " + str(datetime.datetime.now().time()) +
                  " -- Training Minibatch: Loss= " + "{:.6f}".format(train_loss[-1]) +
                  " Overall Accuracy= " + "{:.4f}".format(acc) +
                  " Normalized Accuracy= " + "{:.4f}".format(_sum / float(train_loader.dataset.num_classes)) +
                  " F1 Score= " + "{:.4f}".format(f1_s) +
                  " Confusion Matrix= " + np.array_str(conf_m).replace("\n", "")
                  )
            sys.stdout.flush()

    return sum(train_loss) / len(train_loss), _sum / float(train_loader.dataset.num_classes)



def main():
    parser = argparse.ArgumentParser(description='main')
    # general options
    parser.add_argument('--operation', type=str, required=True, help='Operation [Options: Train | Test]')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to to save outcomes (such as images and trained models) of the algorithm.')

    # dataset options
    parser.add_argument('--img_dir', type=str, required=False, help='Dataset path.')
    parser.add_argument('--mask_dir', type=str, required=False, help='Dataset path.')
    
    parser.add_argument('--val_img_dir', type=str, required=False, help='Dataset path.')
    parser.add_argument('--val_mask_dir', type=str, required=False, help='Dataset path.')
    parser.add_argument('--org_mask_dir', type=str, required=False, help='Dataset path.')

    parser.add_argument('--testing_images_path', type=str, required=False, help='Dataset path.')
    parser.add_argument('--testing_images', type=str, nargs="+", required=False, help='Testing image names.')
    
    parser.add_argument('--crop_size', type=int, required=False, help='Crop size.')
    parser.add_argument('--stride_crop', type=int, required=False, help='Stride size')

    # model options
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['deeplab', 'fcnwideresnet'], help='Model to evaluate')
    parser.add_argument('--model_path', type=str, default=None, help='Model path.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epoch_num', type=int, default=500, help='Number of epochs')

    # handling imbalanced data
    parser.add_argument('--loss_weight', type=float, nargs='+', default=[1.0, 1.0], help='Weight Loss.')
    parser.add_argument('--weight_sampler', type=str2bool, default=False, help='Use weight sampler for loader?')
    args = parser.parse_args()
    print(args)

    # Making sure output directory is created.
    pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # writer for the tensorboard
    writer = SummaryWriter(os.path.join(args.output_path, 'logs'))

    if args.operation == 'Train':
        print('---- training data ----')
        train_set = NGTrain(args.img_dir, args.mask_dir, args.output_path)

        #train_set = DataLoader('Train', args.dataset_path, args.training_images, args.crop_size, args.stride_crop,
        #                       args.output_path)
        
        print('---- validation data ----')
        validation_set = NGValid(args.val_img_dir, args.val_mask_dir, args.org_mask_dir, args.crop_size, args.output_path)

        if args.weight_sampler is False:
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
        else:
            class_loader_weights = 1. / np.bincount(train_set.gen_classes)
            samples_weights = class_loader_weights[train_set.gen_classes]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(samples_weights),
                                                                     replacement=True)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                       num_workers=NUM_WORKERS, drop_last=False, sampler=sampler)

        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # Setting network architecture.
        model = model_factory(args.model_name, train_set.num_channels, train_set.num_classes).cuda()

        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(args.loss_weight),
                                        ignore_index=train_set.num_classes).cuda()

        # Setting optimizer.
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                               betas=(0.9, 0.99))
        # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

        curr_epoch = 1
        best_records = []
        if args.model_path is not None:
            print('Loading model ' + args.model_path)
            best_records = np.load(os.path.join(args.output_path, 'best_records.npy'), allow_pickle=True)
            model.load_state_dict(torch.load(args.model_path))
            # optimizer.load_state_dict(torch.load(args.model_path.replace("model", "opt")))
            curr_epoch += int(os.path.basename(args.model_path)[:-4].split('_')[-1])
            for i in range(curr_epoch):
                scheduler.step()
        model.cuda()

        # Iterating over epochs.
        print('---- training ----')
        for epoch in range(curr_epoch, args.epoch_num + 1):
            # Training function.
            t_loss, t_nacc = train(train_loader, model, criterion, optimizer, epoch)
            writer.add_scalar('Train/loss', t_loss, epoch)
            writer.add_scalar('Train/acc', t_nacc, epoch)
            if epoch % VAL_INTERVAL == 0:
                # Computing test.
                acc, nacc, f1_s, kappa, track_cm = validate(validation_loader, model, epoch, args.output_path)
                writer.add_scalar('Test/acc', nacc, epoch)
                save_best_models(model, args.output_path, best_records, epoch, kappa)
                # patch_acc_loss=None, patch_occur=None, patch_chosen_values=None
            scheduler.step()
    elif args.operation == 'Test':
        print('---- testing data ----')
        test_set = DataLoader('Test', args.testing_images_path, args.testing_images, args.crop_size, args.stride_crop,
                              args.output_path)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

        # Setting network architecture.
        model = model_factory(args.model_name, test_set.num_channels, test_set.num_classes).cuda()

        best_records = np.load(os.path.join(args.output_path, 'best_records.npy'), allow_pickle=True)
        index = 0
        for i in range(len(best_records)):
            if best_records[index]['kappa'] < best_records[i]['kappa']:
                index = i
        epoch = int(best_records[index]['epoch'])
        print("loading model_" + str(epoch) + '.pth')
        model.load_state_dict(torch.load(os.path.join(args.output_path, 'model_' + str(epoch) + '.pth')))
        model.cuda()

        test_full_map(test_loader, model, epoch, args.output_path)
    else:
        raise NotImplementedError("Process " + args.operation + "not found!")


if __name__ == "__main__":
    main()
