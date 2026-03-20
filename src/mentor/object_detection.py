import torchvision
import yolov5
import torch

from .util import current_epoch
from .evaluation import TormetingEvaluator
import tqdm



def create_object_detection_model(arch, n_classes, pretrained=True, freeze_layers_before=0, device="cuda", class_names=""):
        if arch.lower() == "yolov5n":
                net=yolov5
                net.fc=torch.nn.Linear(in_features=2048, out_features=n_classes, bias=True)
        elif arch.lower() == "mobilenetv3":
                #len(list(net.parameters())) == 142
                #len(list(net.classifier.parameters())) == 4
                net = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
                net.classifier[-1]=torch.nn.Linear(in_features=1024, out_features=n_classes, bias=True)
        else:
                raise ValueError
        for param in list(net.parameters())[:freeze_layers_before]:
                param.requires_grad = False
        net = net.to(device)
        net.status = (0, 0., 0.) # Epoch, Validation Error, Train Error
        if class_names == "":
                net.class_names = () #tuple([f"cl_{n}" for n in range(n_classes)])
        else:
                class_names = tuple(class_names.split("\n"))
                if len(net.class_names) != 0 and class_names != net.clas_names:
                        assert net.class_names == class_names
        net.args_history = {}
        net.train_history = []
        net.validation_history = {}
        net.best_weights = net.state_dict()
        return net
