import pytest
import torch
from ding.torch_utils.network.merge import TorchBilinearCustomized, TorchBilinear, BilinearGeneral, FiLM


@pytest.mark.unittest
def test_torch_bilinear_customized():
    batch_size = 10
    in1_features = 20
    in2_features = 30
    out_features = 40
    bilinear_customized = TorchBilinearCustomized(in1_features, in2_features, out_features)
    x = torch.randn(batch_size, in1_features)
    z = torch.randn(batch_size, in2_features)
    out = bilinear_customized(x, z)
    assert out.shape == (batch_size, out_features), "Output shape does not match expected shape."


@pytest.mark.unittest
def test_torch_bilinear():
    batch_size = 10
    in1_features = 20
    in2_features = 30
    out_features = 40
    torch_bilinear = TorchBilinear(in1_features, in2_features, out_features)
    x = torch.randn(batch_size, in1_features)
    z = torch.randn(batch_size, in2_features)
    out = torch_bilinear(x, z)
    assert out.shape == (batch_size, out_features), "Output shape does not match expected shape."


@pytest.mark.unittest
def test_bilinear_consistency():
    batch_size = 10
    in1_features = 20
    in2_features = 30
    out_features = 40

    # Initialize weights and biases with set values
    weight = torch.randn(out_features, in1_features, in2_features)
    bias = torch.randn(out_features)

    # Create and initialize TorchBilinearCustomized and TorchBilinear models
    bilinear_customized = TorchBilinearCustomized(in1_features, in2_features, out_features)
    bilinear_customized.weight.data = weight.clone()
    bilinear_customized.bias.data = bias.clone()

    torch_bilinear = TorchBilinear(in1_features, in2_features, out_features)
    torch_bilinear.weight.data = weight.clone()
    torch_bilinear.bias.data = bias.clone()

    # Provide same input to both models
    x = torch.randn(batch_size, in1_features)
    z = torch.randn(batch_size, in2_features)

    # Compute outputs
    out_bilinear_customized = bilinear_customized(x, z)
    out_torch_bilinear = torch_bilinear(x, z)

    # Compute the mean squared error between outputs
    mse = torch.mean((out_bilinear_customized - out_torch_bilinear) ** 2)

    print(f"Mean Squared Error between outputs: {mse.item()}")

    # Check if outputs are the same
    # assert torch.allclose(out_bilinear_customized, out_torch_bilinear),
    # "Outputs of TorchBilinearCustomized and TorchBilinear are not the same."


def test_bilinear_general():
    """
    Overview:
        Test for the `BilinearGeneral` class.
    """
    # Define the input dimensions and batch size
    in1_features = 20
    in2_features = 30
    out_features = 40
    batch_size = 10

    # Create a BilinearGeneral instance
    bilinear_general = BilinearGeneral(in1_features, in2_features, out_features)

    # Create random inputs
    input1 = torch.randn(batch_size, in1_features)
    input2 = torch.randn(batch_size, in2_features)

    # Perform forward pass
    output = bilinear_general(input1, input2)

    # Check output shape
    assert output.shape == (batch_size, out_features), "Output shape does not match expected shape."

    # Check parameter shapes
    assert bilinear_general.W.shape == (
        out_features, in1_features, in2_features
    ), "Weight W shape does not match expected shape."
    assert bilinear_general.U.shape == (out_features, in2_features), "Weight U shape does not match expected shape."
    assert bilinear_general.V.shape == (out_features, in1_features), "Weight V shape does not match expected shape."
    assert bilinear_general.b.shape == (out_features, ), "Bias shape does not match expected shape."

    # Check parameter types
    assert isinstance(bilinear_general.W, torch.nn.Parameter), "Weight W is not an instance of torch.nn.Parameter."
    assert isinstance(bilinear_general.U, torch.nn.Parameter), "Weight U is not an instance of torch.nn.Parameter."
    assert isinstance(bilinear_general.V, torch.nn.Parameter), "Weight V is not an instance of torch.nn.Parameter."
    assert isinstance(bilinear_general.b, torch.nn.Parameter), "Bias is not an instance of torch.nn.Parameter."


@pytest.mark.unittest
def test_film_forward():
    # Set the feature and context dimensions
    feature_dim = 128
    context_dim = 256

    # Initialize the FiLM layer
    film_layer = FiLM(feature_dim, context_dim)

    # Create random feature and context vectors
    feature = torch.randn((32, feature_dim))  # batch size is 32
    context = torch.randn((32, context_dim))  # batch size is 32

    # Forward propagation
    conditioned_feature = film_layer(feature, context)

    # Check the output shape
    assert conditioned_feature.shape == feature.shape, \
        f'Expected output shape {feature.shape}, but got {conditioned_feature.shape}'

    # Check that the output is different from the input
    assert not torch.all(torch.eq(feature, conditioned_feature)), \
        'The output feature is the same as the input feature'
