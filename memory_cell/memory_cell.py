import torch
import torch.nn as nn

class MemoryCell(nn.Module):

    def __init__(self, input_size, hidden_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sig_moid = nn.LogSigmoid()
        self.tanh = nn.Tanh()
        # input gate parameters
        self.w_ix = nn.Parameter(torch.rand(input_size, hidden_size))
        self.w_ic = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.w_im = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.rand(hidden_size))
        # forget gate
        self.w_fx = nn.Parameter(torch.rand(input_size, hidden_size))
        self.w_mf = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.w_cf = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.rand(hidden_size))
        # output gate
        self.w_ox = nn.Parameter(torch.rand(input_size, hidden_size))
        self.w_om = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.w_oc = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.rand(hidden_size))
        # hidden state params
        self.w_cx = nn.Parameter(torch.rand(input_size, hidden_size))
        self.w_cm = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.rand(hidden_size))

    def forward(self, x):
        c_0 = torch.rand(x.shape[0], self.hidden_size)
        m_0 = torch.rand(x.shape[0], self.hidden_size)

        i_t = self.sig_moid(x @ self.w_ix + m_0 @ self.w_im +
                            c_0 @ self.w_ic + self.b_i)
        f_t = self.sig_moid(x @ self.w_fx + m_0 @ self.w_mf +
                            c_0 @ self.w_cf + self.b_f)
        c_0 = f_t * c_0 + i_t * self.tanh(x @ self.w_cx +
                                          m_0 @ self.w_cm + self.b_c)
        o_t = self.sig_moid(x @ self.w_ox + m_0 @ self.w_om +
                            c_0 @ self.w_oc + self.b_o)
        m_0 = o_t * self.tanh(c_0)
        return m_0