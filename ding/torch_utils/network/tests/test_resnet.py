import pytest
import torch
from ding.torch_utils.network import resnet18
from ding.torch_utils.network.resnet \
    import ResNet, BasicBlock, Bottleneck, AvgPool2dSame, avg_pool2d_same, ClassifierHead
from itertools import product


@pytest.mark.unittest
def test_resnet18():
    model = resnet18()
    print(model)
    inputs = torch.randn(4, 3, 224, 224)
    outputs = model(inputs)
    assert outputs.shape == (4, 1000)


stem_type = ['', 'deep', 'deep,tiered']
replace_stem_pool = [True, False]
avg_down = [True, False]
block = [BasicBlock]
layers = [[2, 2, 2, 2]]
zero_init_last_bn = [True, False]
output_stride = [8, 32]
num_classes = [0, 1000]
args = [
    item for item in
    product(*[stem_type, replace_stem_pool, avg_down, block, layers, zero_init_last_bn, output_stride, num_classes])
]


@pytest.mark.unittest
@pytest.mark.parametrize(
    'stem_type, replace_stem_pool, avg_down, block, layers, zero_init_last_bn, output_stride, num_classes', args
)
def test_resnet(stem_type, replace_stem_pool, avg_down, block, layers, zero_init_last_bn, output_stride, num_classes):
    model = ResNet(
        stem_type=stem_type,
        replace_stem_pool=replace_stem_pool,
        avg_down=avg_down,
        block=block,
        layers=layers,
        output_stride=output_stride,
        num_classes=num_classes,
        drop_rate=0. if stem_type == 'deep' else 0.05
    )
    model.init_weights(zero_init_last_bn=zero_init_last_bn)
    inputs = torch.randn(4, 3, 224, 224).requires_grad_(True)
    outputs = model(inputs)
    assert outputs.shape == (4, num_classes if num_classes > 0 else 512)
    mse_loss = torch.nn.MSELoss()
    target = torch.randn(outputs.shape)
    loss = mse_loss(outputs, target)
    assert inputs.grad is None
    loss.backward()
    assert isinstance(inputs.grad, torch.Tensor)

    model.reset_classifier(num_classes=183)
    inputs = torch.randn(4, 3, 224, 224).requires_grad_(True)
    outputs = model(inputs)
    assert outputs.shape == (4, 183)
    target = torch.randn(outputs.shape)
    loss = mse_loss(outputs, target)
    assert inputs.grad is None
    loss.backward()
    assert isinstance(inputs.grad, torch.Tensor)

    clf = model.get_classifier()
    outputs = model.forward_features(x=inputs)


@pytest.mark.unittest
def test_avg_pool2d_same():
    x = torch.randn(4, 4, 4, 4).requires_grad_(True)
    avg_pool2d_same(x=x, kernel_size=(2, 2), stride=(2, 2))


inplanes = [4]
planes = [1]
args_btl = [item for item in product(*[inplanes, planes])]


@pytest.mark.unittest
@pytest.mark.parametrize('inplanes, planes', args_btl)
def test_Bottleneck(inplanes, planes):
    model = Bottleneck(inplanes=inplanes, planes=planes)
    x = torch.randn(4, 4, 4, 4).requires_grad_(True)
    outputs = model(x)
    assert outputs.shape == (4, 4, 4, 4)
    model.zero_init_last_bn()


in_chs = [1]
num_classes = [0, 1]
drop_rate = [0, 0.05]
args_cls = [item for item in product(*[in_chs, num_classes, drop_rate])]


@pytest.mark.unittest
@pytest.mark.parametrize('in_chs, num_classes, drop_rate', args_cls)
def test_ClassifierHead(in_chs, num_classes, drop_rate):
    model = ClassifierHead(in_chs=in_chs, num_classes=num_classes, drop_rate=drop_rate)
    inputs = torch.randn(1, 1, 1, 1).requires_grad_(True)
    outputs = model(inputs)
    assert outputs.shape == (1, 1, 1, 1)
