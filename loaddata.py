import numpy as np
import cv2
# from dataset import blur


def load_data(dataset_root='./dataset/N19_align.npz'):
	print('======================')
	print('data loading...')
	dataset = np.load(dataset_root, allow_pickle=True)

	return dataset


if __name__ == "__main__":
    load_data()