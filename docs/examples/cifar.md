# CIFAR-10 with ResNet

The `examples/cifar/train_cifar.py` script trains a torchvision ResNet on
CIFAR-10 using a {class}`~mentor.Mentee` subclass.  It demonstrates the full
mentor workflow: argument parsing, training, validation, checkpointing, and
resuming.

## Running the example

```bash
python examples/cifar/train_cifar.py \
    -resume_path cifar.pt \
    -epochs 30 \
    -batch_size 64 \
    -pseudo_batch 2 \
    -lr 0.001 \
    -resnet resnet18 \
    -pretrained true \
    -device cuda \
    -verbose true
```

Resuming a run simply re-uses the same `-resume_path`:

```bash
python examples/cifar/train_cifar.py -resume_path cifar.pt -epochs 60
```

## Key design decisions

**Single path for save and load**
: `-resume_path` is used both for loading an existing checkpoint and for
  writing each new checkpoint.  If the file does not exist, training starts
  from scratch.

**BatchNorm in eval mode during training**
: CIFAR batches can be as small as 1 sample (e.g. the last batch with
  `drop_last=False`), which would cause BatchNorm to fail.
  `CifarResNet` overrides {meth}`~torch.nn.Module.train` to call `m.eval()`
  on every BatchNorm layer after `super().train()`.

**Gradient accumulation**
: `pseudo_batch` accumulates gradients over several mini-batches before
  calling `optimizer.step()`, allowing a larger effective batch size without
  increasing GPU memory.

**Inference state**
: The CIFAR class names are registered with
  {meth}`~mentor.Mentee.register_inference_state` so any checkpoint is
  self-contained for inference.

## Source walkthrough

```{literalinclude} ../../examples/cifar/train_cifar.py
:language: python
:linenos:
```
