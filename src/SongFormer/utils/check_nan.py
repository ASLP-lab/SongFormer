import torch

class NanInfError(Exception):
    """Exception raised for errors in the input."""

    def __init__(self, message, code=None):

        super().__init__(message)
        self.code = code

    def __str__(self):
        if self.code:
            return f"[Error {self.code}] {self.args[0]}"
        return self.args[0]


def check_tensor_nan(tensor):
    if torch.isnan(tensor).any():
        print("NaN detected")
        raise NanInfError("NaN detected")


def check_model_grad(model, step):
    nan_lists = []
    inf_lists = []

    for name, param in model.named_parameters():
        if param.grad is not None and (torch.isnan(param.grad).any()):
            nan_lists.append(name)
        if param.grad is not None and (torch.isinf(param.grad).any()):
            inf_lists.append(name)
    if nan_lists or inf_lists:
        raise NanInfError(
            f"🚨 [Step {step}]\n Parameter `{nan_lists}` became NaN after optimizer step!\nParameter `{inf_lists} became Inf after optimizer step!`"
        )


def check_model_param(model, step):
    nan_lists = []
    inf_lists = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_lists.append(name)
        if torch.isinf(param).any():
            inf_lists.append(name)
    if nan_lists or inf_lists:
        raise NanInfError(
            f"🚨 [Step {step}]\n Parameter `{nan_lists}` became NaN after optimizer step!\nParameter `{inf_lists} became Inf after optimizer step!`"
        )