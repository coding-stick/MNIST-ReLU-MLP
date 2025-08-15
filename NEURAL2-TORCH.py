import pygame
import numpy as np
from pygame.locals import *
import pandas as pd
import pandas as pd
from Objects import Box, Neuron, Connection


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('mnist_mapped.csv', index_col=0)

label = data['label']
features = data.drop(columns=['label'])

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2)
#Convert to Pytorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)


train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # logits
        return x

model = MNISTNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


pygame.init()




# ---- Config ----
WIDTH, HEIGHT = 1200, 800
LEARNING_RATE = 0.1
INPUT_SIZE = 28  # example grid size
TARGET = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # must match output size
NEURONS_PER_LAYER = [model.fc1.in_features, model.fc2.in_features, model.fc3.in_features, len(TARGET)]  # input (incl. bias), hidden, output


mode = "stopped"

win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network Visualizer")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)





# ---- Build grid ----
GRID_AREA_SIZE = WIDTH // 4  # allocate left 1/4 of screen for grid

# Calculate max box size based on available space and input size (with spacing)
BOX_SPACING = INPUT_SIZE **0.2
BOX_LENGTH = (GRID_AREA_SIZE- (INPUT_SIZE * BOX_SPACING)) / INPUT_SIZE

BOARD_POS = (0, HEIGHT * 0.4)  # position grid lower-left-ish

# Build boxes with dynamic size and spacing
box_list = []
for row in range(INPUT_SIZE):
    for col in range(INPUT_SIZE):
        x = BOARD_POS[0] + col * (BOX_LENGTH + BOX_SPACING)
        y = BOARD_POS[1] + row * (BOX_LENGTH + BOX_SPACING)
        box_list.append(Box((x, y), BOX_LENGTH))




# ---- Build network ----
neural = []
connections = []

layer_spacing = WIDTH // (len(NEURONS_PER_LAYER) + 1)
for layer_i, num_neurons in enumerate(NEURONS_PER_LAYER):
    layer = []
    y_spacing = HEIGHT // (num_neurons + 1)
    for n in range(num_neurons):
        bias = (layer_i < len(NEURONS_PER_LAYER) - 1 and n == num_neurons - 1)  # bias neuron in all but last layer
        neuron = Neuron(
            x=WIDTH/8 + (layer_spacing * (layer_i + 1)),
            y=y_spacing * (n + 1),
        )
        layer.append(neuron)
    neural.append(layer)


# Build connections
for layer_i in range(1, len(neural)):
    layer_conns = []
    for prev_neuron in neural[layer_i - 1]:
        for curr_neuron in neural[layer_i]:
            layer_conns.append(Connection(prev_neuron, curr_neuron))
    connections.append(layer_conns)



def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum()).tolist()





# ---- Main loop ----
running = True
batch_iterator = iter(train_loader)
output = None
target = None
batch_count = 0
correct = 0
while running:
    win.fill((0, 0, 0))

    if mode == "train":
        try:
            data, target = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(train_loader)
            data, target = next(batch_iterator)
        
        
        # Show first sample in batch on pixel boxes
        for b, box in enumerate(box_list):
            box.val = data[0][b].item()  # data is tensor batch, take first sample
        
        # Train on batch
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Access weights and draw
        # List of PyTorch weight tensors in order of layers:
        weight_matrices = [model.fc1.weight, model.fc2.weight, model.fc3.weight]

        for layer_idx, weight_tensor in enumerate(weight_matrices):
            weights = weight_tensor.detach().cpu().numpy()  # shape (to_neurons, from_neurons)
            weights_T = weights.T  # shape (from_neurons, to_neurons)

            num_from = weights_T.shape[0]
            num_to = weights_T.shape[1]


            # Assign weights to corresponding connections
            for i in range(num_from):
                for j in range(num_to):
                    idx = i * num_to + j  # index in connections[layer_idx]
                    connections[layer_idx][idx].val = weights_T[i, j]


            for conn in connections[layer_idx]:
                conn.draw(win)

        batch_count += 1

    for layer in connections:
        for conn in layer:
            conn.draw(win)

    # Draw neurons
    for l, layer in enumerate(neural):
        if l==len(neural)-1:
            if output!=None:
                predictions = softmax(output.detach().cpu().numpy()[0].tolist())
                highest_n = predictions.index(max(predictions))

                for n_id, neuron in enumerate(layer):
                    if n_id == highest_n:
                        neuron.draw(win, color=(255, 0, 0))
                        win.blit(font.render(f"{predictions[n_id]:.2f}", True, (255, 0,0)), (neuron.x+10, neuron.y))
                    else:
                        neuron.draw(win)
                        win.blit(font.render(f"{predictions[n_id]:.2f}", True, (255, 255, 255)), (neuron.x+10, neuron.y))
                win.blit(font.render(f"Prediction: {highest_n}", True, (255, 255, 255)), (0, 40))
            else:
                predictions = [0]
        else:
            for neuron in layer:
                neuron.draw(win)




    win.blit(font.render(f"Mode: {mode.upper()}, press T to pause/continue", True, (255, 255, 255)), (0, 10))
    if target !=None:
        win.blit(font.render(f"Target: {int(target[0].item())}", True, (255, 255, 255)), (0, 60))
        if int(target[0].item()) == highest_n and mode=="train":
            correct +=1

    if batch_count!=0:
        win.blit(font.render(f"ACCURACY: {((correct/batch_count)*100):.2f}% ({correct}/{batch_count})", True, (255, 255, 255)), (0, 100))
    win.blit(font.render(f"Batch count: {batch_count}", True, (255, 255, 255)), (0, 80))


    for box in box_list:
        box.draw(win)
    pygame.display.flip()
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                mode = "stopped" if mode == "train" else "train"

pygame.quit()
