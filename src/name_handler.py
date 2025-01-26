def build_model_path(simple_name, appliance, regularizer):
    if regularizer is None:
        kernel_reg = 'none'
    else:
        kernel_reg = regularizer
    return '../models/{}_{}_{}'.format(simple_name, appliance, kernel_reg)
