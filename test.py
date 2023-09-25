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

	model = LFIClassification()
	loss_fn = nn.CrossEntropyLoss()

	model.load_state_dict(torch.load("checkpoints/model.pt"), strict=False)
	model = model.to(device)

	imgs, labels = data_reader.read_in_batch()
	imgs = torch.from_numpy(np.moveaxis(imgs, 5, 1))
	labels = torch.from_numpy(labels)

	test(device, imgs, labels, model, loss_fn)


def test(device, X, y, model, loss_fn):
	X = X.to(device=device, dtype=torch.float)
	pred = model(X)

	print(y)
	print(pred)
	output = pred.cpu().detach().numpy()
	print(output)
	print(np.argmax(output, axis=1))


if __name__ == "__main__":
	main()