# Understanding LSTM Architecture and Components

LSTMs (Long Short-Term Memory) are known for their ability to learn long-term dependencies. An LSTM cell consists of three gates (forget gate, input gate, output gate) and a cell state.

## Contents

- [Cell State](#cell-state)
- [Forget Gate](#forget-gate)
- [Input Gate](#input-gate)
- [Output Gate](#output-gate)
- [LSTM Operation Steps](#lstm-operation-steps)
- [Code Examples](#code-examples)
- [Importance of LSTM Parameters](#importance-of-lstm-parameters)

## Cell State

The cell state carries information and enables the learning of long-term dependencies.

## Forget Gate

The forget gate determines what information should be discarded from the cell state.

```python
import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), 1)
        f_t = self.sigmoid(self.forget_gate(combined))
        c_t = f_t * c_prev
        return h_prev, c_t
