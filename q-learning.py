import numpy as np
import pandas as pd
import time
import math
epsilon = 0.5
gamma = 0.98
lr = 0.2

itemList = \
    [[0, 0.0, 'Vijayawada', 19.4, 15.1, 0, 0, 0.0],
 [1, 1.9323814116266074, 'Tanguturu', 14.5, 5.3, 6600, 2, 1.0],
 [2, 20.857190790708088, 'Podili', 10.7, 7.0, 24000, 2, 0.5],
 [3, 33.077041744388232, 'Ongole', 14.5, 6.2, 305000, 4, 2.5],
 [4, 33.077041744388232, 'Markapur', 7.7, 8.2, 60000, 2, 0.5],
 [5, 34.486376839557948, 'KaniGiri', 9.6, 5.1, 24000, 2, 2.5],
 [6, 34.486376839557948, 'Kondukur', 13.2, 3.5, 90000, 2, 1.0],
 [7, 35.762110362784796, 'Giddalur', 3.8, 5.0, 25000, 2, 1.0],
 [8, 35.762110362784796, 'Chirala', 17.2, 9.0, 98000, 4, 2.0],
 [9, 41.826126709510191, 'Bestavapetta', 6.3, 6.3, 25000, 2, 0.5],
 [10, 46.397278485704312, 'Addanki', 13.9, 8.8, 60000, 2, 0.5],
 [11, 47.372912618077464, 'Chilakalurupet', 15.4, 11.4, 92000, 2, 1.0],
 [12, 59.32442405620133, 'Narasaraopet', 14.5, 12.5, 100000, 4, 1.0],
 [13, 62.987927732225053, 'Vinukonda', 11.8, 11.0, 65000, 2, 1.0],
 [14, 62.987927732225053, 'Tadikonda', 18.1, 14.3, 60000, 2, 1.0],
 [15, 66.744129468890392, 'Sattenapalle', 15.2, 14.0, 45000, 2, 1.0],
 [16, 68.538275316497419, 'Repalie', 21.3, 10.6, 50000, 2, 1.0],
 [17, 73.544821959944969, 'Guntur', 18.0, 13.0, 450000, 4, 3.0],
 [18, 73.544821959944969, 'Vuyyuru', 21.3, 13.6, 39000, 4, 1.0],
 [19, 74.453128626270612, 'Tenali', 19.7, 12.5, 140000, 4, 1.0],
 [20, 75.585513278670021, 'Pamarru', 22.3, 13.2, 62000, 2, 1.0],
 [21, 75.795182234229088, 'Nuzvid', 21.3, 17.5, 37000, 2, 0.5],
 [22, 75.795182234229088, 'Machilipatnam', 23.8, 12.0, 108000, 4, 1.0],
 [23, 77.065444537483828, 'Kaikalur', 24.4, 15.5, 48000, 2, 1.0],
 [24, 88.605535249215663, 'Jaggayyapeta', 14.9, 18.5, 37000, 2, 0.5],
 [25, 88.605535249215663, 'HanumenJunction', 19.5, 15.2, 50000, 2, 1.0],
 [26, 89.19358507516111, 'Gudivada', 22.7, 14.3, 180000, 2, 1.0],
 [27, 89.19358507516111, 'Bapatia', 18.2, 9.7, 82000, 2, 1.0],
 [28, 107.98018686407246, 'Rajahmundry', 29.5, 19.6, 470000, 4, 3.5],
 [29, 107.98018686407246, 'Mandapeta', 30.8, 18.3, 170000, 2, 2.0],
 [30, 114.27222071107218, 'Narasapur', 28.7, 14.5, 160000, 2, 1.0],
 [31, 117.99400024882618, 'Amaiapuram', 31.5, 15.6, 90000, 2, 1.0],
 [32, 119.69071254729833, 'Kakinada', 33.5, 19.1, 228000, 4, 2.0],
 [33, 119.69071254729833, 'Kovvur', 29.0, 19.7, 45000, 2, 1.0],
 [34, 127.33938989016715, 'Tanuku', 28.8, 17.4, 134000, 2, 1.0],
 [35, 132.23053168765526, 'Nidavole', 28.5, 18.7, 50000, 2, 1.0],
 [36, 133.71883894919219, 'Tadepallegudem', 27.7, 17.9, 130000, 4, 1.5],
 [37, 138.82247427963529, 'Eluru', 23.6, 17.0, 198000, 4, 2.0],
 [38, 138.82247427963529, 'Palakolu', 25.9, 15.7, 180000, 4, 1.0],
 [39, 145.45583114719054, 'Bhimavaram', 7.3, 15.3, 148000, 4, 1.5]]

#
def getDistance(a, b, item=itemList):
    """
    get the distance between two points
    :param a: the first point
    :param b: the second point
    :param item: the itemList
    :return: the distance(float) between point a and point b
    """
    x1 = item[a][3]
    y1 = item[a][4]
    x2 = item[b][3]
    y2 = item[b][4]
    return math.sqrt((x1-x2)**2+(y1-y2)**2)



dis = []
for i in range(len(itemList)):
    for j in range(len(itemList)):
        dis.append(getDistance(i,j))
distances = np.array(dis).reshape((40, 40))* 12.2 * 1.12
print(distances)
print(np.max(np.max(distances, axis=1)))
#distance =np.array([[0,7,6,1,3],[7,0,3,7,8],[6,3,0,12,11],[1,7,12,0,2],[3,8,11,2,0]])
#R_table = 11-distance
R_table=np.max(np.max(distances, axis=1))-distances
#print(distance)

space = np.arange(0,40)
print(space)
Q_table = np.zeros((40,40))
print(R_table.shape)
print(Q_table[0])






# 构建Q表格
def build_q_table(n_states, actions):
	table = pd.DataFrame(
		np.zeros((n_states, len(actions))),
		columns=actions,
	)
	return table

# 选择行为
def choose_action(state, q_table):
	state_actions = q_table.iloc[state, :]
	if (((state_actions==0).all())
		or (np.random.uniform() > EPSILON)  # ? 为什么加
		):
		# 两个状态都为零或一个随机概率
		action_name = np.random.choice(ACTIONS)
	else:
		action_name = state_actions.idxmax()
	return action_name
# 获取环境反馈
def get_env_feedback(S, A):
	# agent与环境的交互
	
	return S_, R 
 
# 更新环境
def update_env(S, episode, step_counter):
	env_list = ['-']*(N_STATES-1) + ['T'] # '-------T' 为环境
	if S == 'terminal':
		interaction = 'Episode %s: total_steps = %s'%(episode+1, step_counter)
		print('\r{}'.format(interaction), end='')
		time.sleep(2)
		print('\r                  ', end='')
	else:
		env_list[S] = 'o'
		interaction = ''.join(env_list)
		print('\r{}'.format(interaction), end='')
		time.sleep(FRESH_TIME)
 
def rl():
	q_table = build_q_table(N_STATES, ACTIONS)
	for episode in range(MAX_EPISODES):
		step_counter = 0
		S = 0
		is_terminated = False
		update_env(S, episode, step_counter)
		while not is_terminated:  # 知道到达终点
			A = choose_action(S, q_table)
			S_, R = get_env_feedback(S, A) # 一步状态和回报
 
			q_predict = q_table.loc[S, A]  # 当前状态行为得分
			if S_ != 'terminal':
				# 根据下一步的行为得分最高的计算回报
				# 即如果下一步预测判断更准确，当前状态取得更高分
				q_target = R + GAMMA * q_table.iloc[S_, :].max()  
			else:
				q_target = R
				is_terminated = True
			# 计算当前状态下做出行为A的更新率
			q_table.loc[S, A] += ALPHA * (q_target - q_predict) # 更新
			S = S_ # 移动到下一个状态
 
			update_env(S, episode, step_counter+1)
			step_counter += 1
	return q_table




#即可得到最佳的TSP路径的Q表

if __name__ == '__main__':
	q_table = rl()
	print('\r\nQ-table:\n')
	print(q_table)