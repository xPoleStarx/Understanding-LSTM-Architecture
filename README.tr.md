# LSTM Mimarisi ve Bileşenleri

LSTM'ler (Long Short-Term Memory), uzun süreli bağımlılıkları öğrenme yeteneği ile bilinirler. LSTM hücresi, üç kapıdan (forget gate, input gate, output gate) ve hücre durumundan oluşur.

## İçindekiler

- [Hücre Durumu (Cell State)](#hücre-durumu-cell-state)
- [Unutma Kapısı (Forget Gate)](#unutma-kapısı-forget-gate)
- [Giriş Kapısı (Input Gate)](#giriş-kapısı-input-gate)
- [Çıkış Kapısı (Output Gate)](#çıkış-kapısı-output-gate)
- [LSTM'nin Çalışma Aşamaları](#lstmnin-çalışma-aşamaları)
- [Kod Örnekleri](#kod-örnekleri)
- [LSTM Parametrelerinin Önemi](#lstm-parametrelerinin-önemi)

## Hücre Durumu (Cell State)

Hücre durumu, bilgiyi taşır ve uzun süreli bağımlılıkları öğrenmeyi sağlar.

## Unutma Kapısı (Forget Gate)

Unutma kapısı, hücre durumundan neyin atılacağını belirler.

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

## Giriş Kapısı (Input Gate)

Giriş kapısı, hücre durumuna hangi yeni bilginin ekleneceğini belirler.

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

## Çıkış Kapısı (Output Gate)

Çıkış kapısı, bir sonraki gizli duruma neyin gönderileceğini belirler.

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

## LSTM'nin Çalışma Aşamaları
### Geçmiş Bilgiyi Unutma

```python

f_t = self.sigmoid(self.forget_gate(combined))
c_t = f_t * c_prev
```
### Yeni Bilgiyi Ekleme

```python

i_t = self.sigmoid(self.input_gate(combined))
c_hat_t = self.tanh(self.cell_candidate(combined))
c_t = f_t * c_prev + i_t * c_hat_t
```
### Çıktı Oluşturma

```python

o_t = self.sigmoid(self.output_gate(combined))
h_t = o_t * self.tanh(c_t)
```

### Kod Örnekleri

Aşağıda, bir LSTM hücresinin tam kodu verilmiştir:

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
## LSTM Parametrelerinin Önemi

+    Ağırlık Matrisi (Weight Matrix): Giriş ve çıkış kapıları için ağırlıklar, modelin girdiyi nasıl işleyeceğini belirler. Eğitim sırasında güncellenirler.
+    Bias (Önyargı): Bias terimleri, modelin daha esnek olmasını sağlar.
+    Hücre Durumu (Cell State): Uzun süreli bağımlılıkları öğrenmeyi sağlar.
+    Gizli Durum (Hidden State): Mevcut zaman adımında modelin çıktısını belirler.

Bu doküman, LSTM mimarisini detaylı bir şekilde açıklamayı ve kod örnekleri ile desteklemeyi amaçlamaktadır. Bu kodu ve açıklamaları kullanarak kendi LSTM modelinizi oluşturabilirsiniz.
