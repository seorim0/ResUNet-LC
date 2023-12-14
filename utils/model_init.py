# get architecture
def get_arch(opt):
    """Retrieve the specified neural network architecture based on user's choice."""

    # Extracting relevant configuration options
    arch = opt.arch
    class_num = opt.class_num
    leads = opt.leads

    print('You choose ' + arch + '...')

    # Check if the chosen architecture is 'ResUNet_LC'
    if arch == 'ResUNet_LC':
        # Import the specified architecture from models
        from models import ResUNet_LC
        model = ResUNet_LC(nOUT=class_num)

    elif arch == 'Res1D_MaxPool':
        from models import Res1D_MaxPool
        model = Res1D_MaxPool(nOUT=class_num)
    elif arch == 'Res2D_MaxPool':
        from models import Res2D_MaxPool
        model = Res2D_MaxPool(nOUT=class_num)
    elif arch == 'Res2D_LC':
        from models import Res2D_LC
        model = Res2D_LC(nOUT=class_num)
    elif arch == 'ResU2D_LC':
        from models import ResU2D_LC
        model = ResU2D_LC(nOUT=class_num)
    else:
        # If the architecture is not found, raise an exception
        raise Exception("Arch error!")

    return model


# get trainer and validator (train method)
def get_train_mode(opt):
    """Retrieve the specified training and validation methods based on the user's choice of loss type."""

    loss_type = opt.loss_type

    print('You choose ' + loss_type + 'trainer ...')

    # Check if the chosen loss type is 'base'
    if loss_type == 'base':
        # Importing the training and validation methods for the 'base' loss type
        from .trainer import base_train
        from .trainer import base_valid
        trainer = base_train
        validator = base_valid
    else:
        # If the loss type is not found, raise an exception
        raise Exception("Loss type error!")

    return trainer, validator


# get loss function
def get_loss(opt):
    """Retrieve the specified loss function based on user's choice."""

    loss_oper = opt.loss_oper
    DEVICE = opt.device

    print('You choose ' + loss_oper + ' loss function ...')

    # Check if the chosen loss operation is 'base'
    if loss_oper == 'base':
        # Use Binary Cross-Entropy with Logits as the loss function
        import torch.nn as nn
        bce_loss = nn.BCEWithLogitsLoss().to(DEVICE)
        return bce_loss
    else:
        # If the loss operation type is not found, raise an exception
        raise Exception("Loss type error!")
