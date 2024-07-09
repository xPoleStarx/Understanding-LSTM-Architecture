# Understanding LSTM Architecture and Components

LSTMs (Long Short-Term Memory) are known for their ability to learn long-term dependencies. An LSTM cell consists of three gates (forget gate, input gate, output gate) and a cell state.

## Contents

- [Cell State](#cell-state)
- [Forget Gate](#forget-gate)
- [Input Gate](#input-gate)
- [Output Gate](#output-gate)
- [How LSTM Works](#how-lstm-works)
- [Code Examples](#code-examples)
- [Importance of LSTM Parameters](#importance-of-lstm-parameters)

## Cell State

The cell state carries information and allows the learning of long-term dependencies.

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
```

## Input Gate

The input gate determines what new information should be added to the cell state.

```python


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.cell_candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), 1)
        f_t = self.sigmoid(self.forget_gate(combined))

        i_t = self.sigmoid(self.input_gate(combined))
        c_hat_t = self.tanh(self.cell_candidate(combined))

        c_t = f_t * c_prev + i_t * c_hat_t
        return h_prev, c_t
```

## Output Gate

The output gate determines what information is sent to the next hidden state.

```python


class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.cell_candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), 1)
        f_t = self.sigmoid(self.forget_gate(combined))

        i_t = self.sigmoid(self.input_gate(combined))
        c_hat_t = self.tanh(self.cell_candidate(combined))

        c_t = f_t * c_prev + i_t * c_hat_t

        o_t = self.sigmoid(self.output_gate(combined))
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t
```

## How LSTM Works
### Forgetting the Past Information

```python


f_t = self.sigmoid(self.forget_gate(combined))
c_t = f_t * c_prev
```
### Adding New Information

```python


i_t = self.sigmoid(self.input_gate(combined))
c_hat_t = self.tanh(self.cell_candidate(combined))
c_t = f_t * c_prev + i_t * c_hat_t
```
### Generating Output

```python


o_t = self.sigmoid(self.output_gate(combined))
h_t = o_t * self.tanh(c_t)
```

## Code Examples

Below is the complete code for an LSTM cell:

```python

import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.cell_candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), 1)

        f_t = self.sigmoid(self.forget_gate(combined))
        i_t = self.sigmoid(self.input_gate(combined))
        c_hat_t = self.tanh(self.cell_candidate(combined))
        c_t = f_t * c_prev + i_t * c_hat_t
        o_t = self.sigmoid(self.output_gate(combined))
        h_t = o_t * self.tanh(c_t)

        return h_t, c_t

input_dim = 10
hidden_dim = 20
lstm_cell = LSTMCell(input_dim, hidden_dim)

x = torch.randn(1, input_dim)
h_prev = torch.zeros(1, hidden_dim)
c_prev = torch.zeros(1, hidden_dim)
hidden = (h_prev, c_prev)

h_next, c_next = lstm_cell(x, hidden)
print(h_next, c_next)
```

## Importance of LSTM Parameters

+   Weight Matrix: Weights for the input and output gates determine how the input is processed. They are updated during training.
+    Bias: Bias terms allow the model to be more flexible.
+    Cell State: Enables learning long-term dependencies.
+    Hidden State: Determines the output of the model at the current time step.

This document aims to explain the LSTM architecture in detail and support it with code examples. Using this code and explanations, you can create your own LSTM model.
