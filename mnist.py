import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import csv
import time
import logging
from datasets import load_dataset


# 로깅 설정: 프로그램이 실행되는 동안 정보를 출력하는 설정입니다.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='./logs/mnist.log')

# 콘솔 핸들러 추가
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# 시간 측정을 위한 변수들: 각 과정에 걸린 시간을 저장합니다.
start_time = time.time()
times = {}

# 데이터 전처리: 이미지를 텐서(숫자로 이루어진 배열)로 변환하고, 값을 정규화합니다.
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# MNIST 데이터셋을 불러옵니다: 손글씨 숫자 이미지 데이터입니다.
ds = load_dataset("ylecun/mnist")

# 데이터셋을 텐서로 변환하고 정규화하는 함수입니다.
def transform_batch(batch):
    images = batch['image']
    labels = batch['label']
    images = [transform(img) for img in images]
    return torch.stack(images), torch.tensor(labels)

# 데이터를 DataLoader로 변환합니다.
train_data = ds['train']
test_data = ds['test']

train_images, train_labels = transform_batch(train_data)
test_images, test_labels = transform_batch(test_data)

trainloader = torch.utils.data.DataLoader(list(zip(train_images, train_labels)), batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(list(zip(test_images, test_labels)), batch_size=32, shuffle=False)

# 모델 정의: 신경망 모델을 만듭니다.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()  # 이미지를 일렬로 펴줍니다.
        self.fc1 = nn.Linear(28 * 28, 128)  # 28x28 크기의 이미지를 128개의 뉴런으로 연결합니다.
        self.fc2 = nn.Linear(128, 10)  # 128개의 뉴런을 10개의 출력으로 연결합니다.

    def forward(self, x):
        x = self.flatten(x)  # 이미지를 일렬로 펼칩니다.
        x = torch.relu(self.fc1(x))  # 첫 번째 레이어를 통과시키고 활성화 함수(ReLU)를 적용합니다.
        x = self.fc2(x)  # 두 번째 레이어를 통과시킵니다.
        return x

model = Net()  # 모델을 생성합니다.

# 손실 함수와 옵티마이저 정의: 모델이 얼마나 틀렸는지 계산하고, 학습을 도와줍니다.
criterion = nn.CrossEntropyLoss()  # 분류 문제에 사용하는 손실 함수입니다.
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 옵티마이저를 사용합니다.

# 모델 훈련: 모델을 학습시킵니다.
training_start_time = time.time()
num_epochs = 5  # 학습을 5번 반복합니다.
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in trainloader:
        optimizer.zero_grad()  # 이전 단계에서 계산된 기울기를 초기화합니다.
        outputs = model(inputs)  # 모델에 입력 데이터를 넣어 예측값을 얻습니다.
        loss = criterion(outputs, labels)  # 손실을 계산합니다.
        loss.backward()  # 손실에 따라 기울기를 계산합니다.
        optimizer.step()  # 모델의 가중치를 업데이트합니다.
        running_loss += loss.item()
    logging.info(f'[{epoch + 1}] loss: {running_loss / len(trainloader)}')  # 각 에포크의 손실을 출력합니다.
times['training_time'] = time.time() - training_start_time  # 훈련 시간 기록

# 모델 평가: 모델이 얼마나 잘하는지 테스트합니다.
evaluation_start_time = time.time()
correct = 0
total = 0
with torch.no_grad():  # 평가할 때는 기울기를 계산하지 않습니다.
    for inputs, labels in testloader:
        outputs = model(inputs)  # 모델에 입력 데이터를 넣어 예측값을 얻습니다.
        _, predicted = torch.max(outputs.data, 1)  # 가장 높은 값을 가진 예측 결과를 얻습니다.
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 맞춘 개수를 셉니다.
test_acc = correct / total  # 정확도를 계산합니다.
times['evaluation_time'] = time.time() - evaluation_start_time  # 평가 시간 기록

logging.info(f'테스트 정확도: {test_acc}')  # 테스트 정확도 출력

# 테스트 데이터에서 이미지 하나를 선택해 예측해봅니다.
dataiter = iter(testloader)
images, labels = next(dataiter)  # Use the built-in next() function

outputs = model(images)  # 모델에 이미지 데이터를 넣어 예측값을 얻습니다.
_, predicted = torch.max(outputs, 1)
predicted_label = predicted[0].item()  # 첫 번째 이미지의 예측 결과를 얻습니다.
logging.info(f'예측한 숫자: {predicted_label}')  # 예측한 숫자 출력
# 훈련 과정의 정확도와 손실 그래프를 출력합니다.
training_losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    training_losses.append(running_loss / len(trainloader))

# 전체 소요 시간 기록
total_time = time.time() - start_time  # 전체 소요 시간을 계산합니다.
logging.info(f'전체 소요 시간: {total_time} 초')  # 전체 소요 시간 출력
with open('./csv/mnist_result.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Total Time', total_time])

# 두 개의 플롯을 함께 표시합니다.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 손실 그래프를 그립니다.
ax1.plot(training_losses, label='loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend(loc='lower right')
ax1.set_title('Training Loss')

# 테스트 데이터에서 이미지 하나를 선택해 예측해봅니다.
ax2.imshow(images[0].numpy().squeeze(), cmap='gray')
ax2.set_title(f'Predicted: {predicted_label}')
ax2.axis('off')

plt.show()

# 결과를 CSV 파일에 저장합니다.
with open('mnist_pytorch_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Step', 'Time (seconds)'])
    for step, time_taken in times.items():
        writer.writerow([step, time_taken])
    writer.writerow(['Test Accuracy', test_acc])
    writer.writerow(['Predicted Label', predicted_label])

