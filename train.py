import cv2
import numpy as np
import torch
from torch import nn
from torchvision.transforms import ToTensor
from data_reader import DataReader
from LFI_classification_model import LFIClassification


def main():
	data_reader = DataReader()
	
	# Get cpu or gpu device for training.
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")

	model = LFIClassification().to(device)
	print(model)

	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

	for i in range(data_reader.epochs):
		imgs, labels = data_reader.read_in_batch()
		imgs = torch.from_numpy(np.moveaxis(imgs, 5, 1))
		labels = torch.from_numpy(labels)
		for j in range(50):
			train(device, imgs, labels, model, loss_fn, optimizer)

	torch.save(model.state_dict(), 'checkpoints/model.pt')


def train(device, X, y, model, loss_fn, optimizer):
	model.train()
	X = X.to(device=device, dtype=torch.float)
	y = y.type(torch.cuda.LongTensor)
	y.to(device)

	# Compute prediction error
	pred = model(X)
	loss = loss_fn(pred, y)

	# Backpropagation
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	loss = loss.item()
	print(f"loss: {loss:>7f}")


if __name__ == "__main__":
	main()