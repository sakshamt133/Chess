# bishop images 0
# knight images 1
# pawn images 2
# rook image 3

import torch
from train_test_data import train_data, test_data
from model import Model
import utils

model = Model(utils.in_channels)
loss = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), utils.lr, weight_decay=0.1)

for epoch in range(utils.epochs):
    sume = 0
    tot = 0
    for (img, labels) in train_data:
        out = model(img)
        real = []
        for val in out:
            real.append(torch.argmax(val))

        for i, val in enumerate(labels):
            if val == real[i]:
                sume += 1
            tot += 1

        l = loss(out, labels.long())
        print(f"for epoch {epoch} loss is {l} accuracy is {sume/tot}")
        l.backward()
        opt.step()
        opt.zero_grad()


with torch.no_grad():
    sume = 0
    tot = 0
    for (img, label) in test_data:
        out = model(img)
        real = []
        for val in out:
            real.append(torch.argmax(val))

        for i, val in enumerate(label):
            if val == real[i]:
                sume += 1
            tot += 1

    print(f'accuracy is {sume / tot}')