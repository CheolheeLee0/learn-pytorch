# 필요한 도구들을 가져와요
import torch  # 숫자를 다루는 특별한 도구
import matplotlib.pyplot as plt  # 그림을 그리는 도구
from mpl_toolkits.mplot3d import Axes3D  # 3D 그림을 그리는 특별한 도구
import logging  # 일기를 쓰는 것처럼 기록을 남기는 도구

# 일기장 설정하기
# 'tensor_log.log'라는 파일에 우리가 한 일을 기록할 거예요
# 시간도 함께 기록해서 언제 무슨 일이 있었는지 알 수 있어요
logging.basicConfig(filename='./logs/visualize_3d.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# 특별한 숫자 모음(텐서)을 만들어요
# 5개의 3x3 크기의 랜덤한 숫자 모음을 만들어요
x = torch.rand(5, 3, 3)

# 1. 만든 숫자 모음을 화면에 보여줘요
print("만들어진 텐서:")
print(x)

# 2. 만든 숫자 모음을 'output.txt' 파일에 저장해요
# 나중에 다시 볼 수 있게 파일로 남겨두는 거예요
with open('output.txt', 'w') as f:
    f.write(str(x))

# 3. 3D 그림으로 숫자 모음을 그려볼 거예요
# 큰 종이(figure)를 준비해요
fig = plt.figure(figsize=(10, 8))
# 3D 그림을 그릴 수 있는 특별한 공간을 만들어요
ax = fig.add_subplot(111, projection='3d')

# 5개의 숫자 모음 각각에 대해 점을 찍어요
for i in range(5):
    # x, y, z 위치를 정해요
    xs = x[i, :, 0].numpy()
    ys = x[i, :, 1].numpy()
    zs = x[i, :, 2].numpy()
    # 정해진 위치에 점을 찍어요
    ax.scatter(xs, ys, zs)

# 그림에 이름표를 달아요
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D 텐서 시각화')

# 그린 그림을 'tensor_visualization_3d.png' 파일로 저장해요
plt.savefig('tensor_visualization_3d.png')
# 그림을 화면에 보여줘요
plt.show()
# 그림 그리기를 끝내요
plt.close()

# 4. 일기장에 우리가 만든 숫자 모음을 기록해요
logging.info(f'만들어진 텐서: {x}')

# 모든 작업이 끝났다고 알려줘요
print("결과물들이 파일로 저장되었어요: output.txt, tensor_visualization_3d.png, 그리고 tensor_log.log")