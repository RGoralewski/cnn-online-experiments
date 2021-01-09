import numpy as np


class CircularQueue:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.tail = -1
        self.head = 0
        self.size = 0

    def push(self, arr: np.ndarray) -> None:
        if self.size == self.capacity:
            print("Error: Queue is full!")
        else:
            self.tail = (self.tail + 1) % self.capacity
            self.queue[self.tail] = arr
            self.size += 1

    def pop(self) -> np.ndarray:
        if self.size == 0:
            print(f"Error: queue is empty!")
            return
        else:
            y = self.queue[self.head]
            self.head = (self.head + 1) % self.capacity
            self.size -= 1
            return y

    def get_window_of_data(self, window_length):
        tmp = []
        for i in range(window_length):
            tmp.append(self.pop())
        return np.concatenate(tmp)

    def get_size(self) -> int:
        return self.size

