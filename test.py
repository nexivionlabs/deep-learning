import torch

# Basit sinir ağı
model = torch.nn.Sequential(
    torch.nn.Linear(2, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1)
)

x = torch.tensor([[1.0, 2.0]])
y = torch.tensor([[1.0]])

loss_fn = torch.nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.01)

for i in range(200):
    pred = model(x)
    loss = loss_fn(pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

print("Tahmin:", pred.item())
