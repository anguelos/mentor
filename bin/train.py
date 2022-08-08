#!/usr/bin/env python3

from copy import copy
import fargv
import torch
import torchvision
import glob
import re
from pathlib import Path
from PIL import Image
import sys
from typing import Union
import sklearn

from types import SimpleNamespace
t_args = Union[SimpleNamespace, dict]


default_transform = torchvision.transforms.Compose([
        torchvision.transforms.transforms.Resize(512), 
        torchvision.transforms.PILToTensor(), 
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def is_tormenting(net: torch.nn.Module):
        """Validates if a torch.nn.Module is a valid tormenting module"""
        return all([field in net.__dict__ for field in ("args_history", "train_history", "validation_history", "best_weights")])

def create_classification_model(archname, n_classes, pretrained=True, freeze_layers_before=0):
        if archname.lower() == "resnet50":
                net=torchvision.models.resnet50(pretrained=pretrained)
                # TODO fc layer
                raise NotImplemented
        elif archname.lower() == "modilenetv3":
                #len(list(net.parameters())) == 142
                #len(list(net.classifier.parameters())) == 4
                net = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
                net.classifier[-1]=torch.nn.Linear(in_features=1024, out_features=n_classes, bias=True)
        else:
                raise ValueError
        for param in list(net.parameters())[:freeze_layers_before]:
                param.requires_grad = False
        net.args_history = {}
        net.train_history = []
        net.validation_history = {}
        net.best_weights = net.state_dict()
        return net


def last(some_dict:dict):
        return some_dict[sorted(some_dict.keys())[-1]]


def current_epoch(net:torch.nn.Module):
        return len(net.train_history)


def warn(*args):
        for item in args:
                sys.stderr.write(str(item))
        sys.stderr.write("\n")
        sys.stderr.flush()


def save(fname,net):
        save_dict = {"weights":net.state_dict(),"args_history":net.args_history, "train_history":net.train_history, "validation_history":net.validation_history, "best_weights":net.best_weights}
        torch.save(save_dict,open(fname,"wb"))


def resume_classification(args, fname):
        net = create_classification_model(args.archname, args.n_classes, args.pretrained, args.freeze_all_before)
        try:
                save_dict = torch.load(open(fname,"rb"))
                new_epoch = len(save_dict["train_history"])
                save_dict["args_history"][new_epoch] = args
                net.args_history = save_dict["args_history"]
                net.train_history = save_dict["train_history"]
                net.validation_history = save_dict["validation_history"]
                net.best_weights = save_dict["best_weights"]
        except FileNotFoundError:
                warn(f"could not load {fname}")


class FolderClassificationDs:
        def __init__(self, file_list, class_names="", class_level=-2, filter_re=None, input_transform=default_transform ) -> None:
                self.class_level=class_level
                if filter_re is not None:
                        regex = re.compile(filter_re)
                        self.files = [f for f in file_list if len(regex.findall(f))>0]
                else:
                        self.files = copy(file_list)
                if class_names is "":
                        self.class_names = sorted(set([f.split("/")[class_level] for f in self.files]))
                else:
                        self.class_names = class_names.split(",")
                        assert all([f.split("/")[class_level] in self.class_names for f in self.files])
                self.class_ids = []
                for image_name in self.files:
                        assert Path(image_name).isfile()
                        self.class_ids.append(self.class_names.index(image_name.split("/")[class_level]))
                if input_transform is None:
                        self.input_transform = lambda x:x
                else:
                        self.input_transform = input_transform


        def __getitem__(self, n:int) -> torch.Tensor:
                img = Image.open(self.files[n])
                return self.input_transform(img), self.class_ids[n]
                

        def __len__(self) -> int:
                return len(self.class_ids)
        
        def __repr__(self):
                return f"FolderClassificationDs({repr(self.files)},{self.class_level},{repr(','.join(self.class_names))})"


class TormetingEvaluator():
        def reset(self):
                pass

        def update(self, *batch_args):
                raise NotImplementedError

        def digest(self)->dict:
                return {"Value":0.}


class TwoClassEvaluator():
        def __init__(self, loss_fn=None, roc_step=.01):
                loss_fn = loss_fn
                self.reset()

        def reset(self):
                self.y_true = []
                self.y_score = []

        def update(self, predictions, targets):
                self.y_score.append(predictions)
                self.y_true.append(targets)
        
        def digest(self):
                result = {}
                y_score = torch.cat(self.predictions, dim=0)
                y_true = torch.cat(self.targets, dim=0)
                if self.loss_fn is not None:
                        with torch.no_grad():
                                losses = self.loss_fn(y_score, y_true).sum()
                        result["loss"] = losses.cpu().numpy()
                y_score, y_train = y_score.cpu().numpy(), y_train.cpu().numpy() 
                roc_auc = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_score)
                accuracy = ((y_score>.5) == y_true).mean()
                result.update({"ROC AUC": roc_auc, "Accuracy": accuracy})
                return result
        
        def __str__(self):
                outputs = self.digest()
                return repr(outputs)


def iterate_classification_epoch(net, dataloader, evaluator:TormetingEvaluator, loss_fn=None, optimizer=None, device="cuda"):
        is_training = optimizer is not None
        if is_training:
                pass # TODO (anguelos) conditional context manager
                net.training(False)
        else:
                net.training(True)
        evaluator.reset()
        for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                if is_training:
                        optimizer.zero_grad()
                        predictions = net(inputs)
                        loss = loss_fn(predictions, targets)
                        loss.backward()
                        optimizer.step()
                else:
                        predictions = net(inputs)
                evaluator.update(predictions, targets)
        if is_training:
                net["train_history"].append(evaluator.digest())
        


if __name__=="__main__":
    p = {
            "data_root":"./data/ingested",
            ""
    }
