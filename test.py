
import pretrainedmodels


net = pretrainedmodels.__dict__['vgg19'](pretrained="imagenet")
print(net)
net = pretrainedmodels.__dict__['resnet50'](pretrained="imagenet")
print(net)