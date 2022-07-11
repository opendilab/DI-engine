import torch


def adjust_ia_state_dict(model_path):
    state_dict = torch.load(model_path)
    for key in list(state_dict['model'].keys()):
        new = key
        if "transformer" in key:
            new = key.replace('layers', 'main').replace('mlp.1', 'mlp.2')
        elif "location_head" in key:
            new = key.replace('GateWeightG', 'gate').replace('UpdateSP', 'update_sp')
        state_dict['model'][new] = state_dict['model'].pop(key)

    torch.save(state_dict, model_path)


if __name__ == '__main__':
    model_path = './tests/rl_model.pth'
    adjust_ia_state_dict(model_path)
