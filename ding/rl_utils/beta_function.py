"""
Referenced papar <Implicit Quantile Networks for Distributional Reinforcement Learning>
"""
import torch
from typing import Union

beta_function_map = {}

beta_function_map['uniform'] = lambda x: x

# For beta functions, concavity corresponds to risk-averse and convexity to risk-seeking policies


# For CPW, eta = 0.71 most closely match human subjects
# this function is locally concave for small values of τ and becomes locally convex for larger values of τ
def cpw(x: Union[torch.Tensor, float], eta: float = 0.71) -> Union[torch.Tensor, float]:
    return (x ** eta) / ((x ** eta + (1 - x) ** eta) ** (1 / eta))


beta_function_map['CPW'] = cpw


# CVaR is risk-averse
def CVaR(x: Union[torch.Tensor, float], eta: float = 0.71) -> Union[torch.Tensor, float]:
    assert eta <= 1.0
    return x * eta


beta_function_map['CVaR'] = CVaR


# risk-averse (eta < 0) or risk-seeking (eta > 0)
def Pow(x: Union[torch.Tensor, float], eta: float = 0.0) -> Union[torch.Tensor, float]:
    if eta >= 0:
        return x ** (1 / (1 + eta))
    else:
        return 1 - (1 - x) ** (1 / 1 - eta)


beta_function_map['Pow'] = Pow
