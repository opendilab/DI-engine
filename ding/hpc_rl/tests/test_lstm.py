import time
import torch
from hpc_rll.origin.rnn import get_lstm
from hpc_rll.torch_utils.network.rnn import LSTM
from testbase import mean_relative_error, times

assert torch.cuda.is_available()
use_cuda = True

seq_len = 64
batch_size = 3
input_size = 1792
hidden_size = 384
num_layers = 3
norm_type = 'LN'
dropout = 0  # 0.1


# Note: need open load_params for hpc_lstm to validation
# Note: only used to case of num_layers = 3
def lstm_val():
    ori_lstm = get_lstm('normal', input_size, hidden_size, num_layers, norm_type, dropout)
    hpc_lstm = LSTM(seq_len, batch_size, input_size, hidden_size, num_layers, norm_type, dropout)

    ori_x = torch.randn(seq_len, batch_size, input_size)
    ori_h0 = torch.randn(num_layers, batch_size, hidden_size)
    ori_c0 = torch.randn(num_layers, batch_size, hidden_size)

    if use_cuda:
        ori_x = ori_x.cuda()
        ori_h0 = ori_h0.cuda()
        ori_c0 = ori_c0.cuda()
        ori_lstm = ori_lstm.cuda()
        hpc_lstm = hpc_lstm.cuda()

    ori_x.requires_grad_(True)
    ori_output, ori_next_state = ori_lstm(ori_x, [ori_h0, ori_c0])
    ori_loss = ori_output.mean()
    ori_loss.backward()

    hpc_x = ori_x.clone().detach()
    hpc_h0 = ori_h0.clone().detach()
    hpc_c0 = ori_c0.clone().detach()
    hpc_x.requires_grad_(True)
    hpc_output, hpc_next_state = hpc_lstm(hpc_x, [hpc_h0, hpc_c0])
    hpc_loss = hpc_output.mean()
    hpc_loss.backward()
    torch.cuda.synchronize()

    mre = mean_relative_error(
        torch.flatten(ori_loss).cpu().detach().numpy(),
        torch.flatten(hpc_loss).cpu().detach().numpy()
    )
    print("lstm fp mean_relative_error: " + str(mre))
    mre = mean_relative_error(
        torch.flatten(ori_x.grad).cpu().detach().numpy(),
        torch.flatten(hpc_x.grad).cpu().detach().numpy()
    )
    print("lstm bp mean_relative_error: " + str(mre))

    ori_wx_grad = torch.cat((ori_lstm.wx[0].grad, ori_lstm.wx[1].grad, ori_lstm.wx[2].grad))
    hpc_wx_grad = hpc_lstm.wx.grad
    mre = mean_relative_error(torch.flatten(ori_wx_grad).cpu().numpy(), torch.flatten(hpc_wx_grad).cpu().numpy())
    print("wx grad mean_relative_error: " + str(mre))

    ori_wh_grad = torch.cat((ori_lstm.wh[0].grad, ori_lstm.wh[1].grad, ori_lstm.wh[2].grad))
    hpc_wh_grad = hpc_lstm.wh.grad
    mre = mean_relative_error(torch.flatten(ori_wh_grad).cpu().numpy(), torch.flatten(hpc_wh_grad).cpu().numpy())
    print("wh grad mean_relative_error: " + str(mre))

    ori_bias_grad = ori_lstm.bias.grad
    hpc_bias_grad = hpc_lstm.bias.grad
    mre = mean_relative_error(torch.flatten(ori_bias_grad).cpu().numpy(), torch.flatten(hpc_bias_grad).cpu().numpy())
    print("bias grad mean_relative_error: " + str(mre))

    params = list(ori_lstm.parameters())
    gamma_0_x = params[1]
    beta_0_x = params[2]
    gamma_0_h = params[3]
    beta_0_h = params[4]
    gamma_1_x = params[5]
    beta_1_x = params[6]
    gamma_1_h = params[7]
    beta_1_h = params[8]
    gamma_2_x = params[9]
    beta_2_x = params[10]
    gamma_2_h = params[11]
    beta_2_h = params[12]
    ori_gamma_grad = torch.cat(
        (gamma_0_x.grad, gamma_0_h.grad, gamma_1_x.grad, gamma_1_h.grad, gamma_2_x.grad, gamma_2_h.grad)
    )
    ori_beta_grad = torch.cat(
        (beta_0_x.grad, beta_0_h.grad, beta_1_x.grad, beta_1_h.grad, beta_2_x.grad, beta_2_h.grad)
    )
    hpc_gamma_grad = hpc_lstm.ln_gamma.grad
    hpc_beta_grad = hpc_lstm.ln_beta.grad
    mre = mean_relative_error(torch.flatten(ori_gamma_grad).cpu().numpy(), torch.flatten(hpc_gamma_grad).cpu().numpy())
    print("ln gamma grad mean_relative_error: " + str(mre))
    mre = mean_relative_error(torch.flatten(ori_beta_grad).cpu().numpy(), torch.flatten(hpc_beta_grad).cpu().numpy())
    print("ln beta grad mean_relative_error: " + str(mre))


def lstm_perf():
    ori_lstm = get_lstm('normal', input_size, hidden_size, num_layers, norm_type, dropout)
    hpc_lstm = LSTM(seq_len, batch_size, input_size, hidden_size, num_layers, norm_type, dropout)

    lstms = {'normal': ori_lstm, 'hpc': hpc_lstm}

    for lstm_type, lstm in lstms.items():
        x = torch.rand(seq_len, batch_size, input_size)
        h0 = torch.randn(num_layers, batch_size, hidden_size)
        c0 = torch.randn(num_layers, batch_size, hidden_size)
        if use_cuda:
            x = x.cuda()
            h0 = h0.cuda()
            c0 = c0.cuda()
            lstm = lstm.cuda()

        prev_state = [h0, c0]
        x.requires_grad_(True)
        for i in range(times):
            t = time.time()
            output, _ = lstm(x, prev_state)
            loss = output.mean()
            loss.backward()
            if use_cuda:
                torch.cuda.synchronize()
            print('epoch: {}, {} lstm cost time: {}'.format(i, lstm_type, time.time() - t))


if __name__ == '__main__':
    print(
        "target problem: seq_len = {}, batch_size = {}, input_size = {}, hidden_size = {}, num_layers = {}, norm_type = {}, dropout = {}"  # noqa
        .format(seq_len, batch_size, input_size, hidden_size, num_layers, norm_type, dropout)
    )
    print("==============lstm has no validation test================")
    #print("===============run lstm validation test==================")
    #lstm_val()
    print("===============run lstm performance test=================")
    lstm_perf()
