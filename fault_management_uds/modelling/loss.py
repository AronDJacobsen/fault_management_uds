import torch


def get_loss(loss_name):
    if loss_name == 'MSELoss':
        #return MSELoss
        return torch.nn.MSELoss(reduction='none')
    elif loss_name == 'MAELoss':
        #return MAELoss
        return torch.nn.L1Loss(reduction='none')
    else:
        raise ValueError(f"Loss {loss_name} not supported.")




def MSELoss(predictions, targets, masks):
    # Calculate the squared differences
    squared_diff = (predictions - targets) ** 2

    # Apply the mask to ignore invalid data
    masked_squared_diff = squared_diff * masks

    # Calculate the mean of the masked squared differences
    mse = torch.sum(masked_squared_diff) / torch.sum(masks)

    return mse

def MAELoss(predictions, targets, masks):
    # Calculate the absolute differences
    abs_diff = torch.abs(predictions - targets)

    # Apply the mask to ignore invalid data
    masked_abs_diff = abs_diff * masks

    # Calculate the mean of the masked absolute differences
    mae = torch.sum(masked_abs_diff) / torch.sum(masks)

    return mae
    