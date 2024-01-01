index = 1234 #0~9999까지의 랜덤 그림

net.eval()
x = mnist_test[index][0].view(28, 28).to(device)
y = mnist_test[index][1]

z = net(x)
pred = torch.max(z, 1)[1].item()

print(f'Predicted: {pred}')
print(f'Label: {y}')

plt.imshow(x.cpu(), cmap='Greys')
plt.show()
