'''

Code for creating and saving n videoclips with the final phase 2 model   

'''

import random
import numpy as np
from collections import namedtuple
import collections
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


# from jaco_env_phase2 import jacoDiverseObjectEnv
from jaco_env_phase2 import jacoDiverseObjectEnv
import pybullet as p


number_episodes = 5

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


STACK_SIZE = 4

'''
DQN
'''
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(STACK_SIZE, 32, kernel_size=7, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w,7,4),5,2),3,2),3,1),3,1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h,7,4),5,2),3,2),3,1),3,1)
        linear_input_size = convw * convh * 64
        self.linear = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.linear(x.view(x.size(0), -1)))
        return self.head(x)

preprocess = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize(128, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_screen(environment):
    global stacked_screens
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    rgb, depth, segmentation = environment._get_observation()
    screen = rgb.transpose((2, 0, 1))
    
    # screen = env._get_observation().transpose((2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)

    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    # Resize, and add a batch dimension (BCHW)
    return preprocess(screen).unsqueeze(0).to(device)


""" Evaluation of trained DQN model"""

PATH = 'models/policyDQN_phase2_bs64_ss4_rb25000_gamma0.99_decaylf20000_lr0.0001.pt'

scores_window = collections.deque(maxlen=100)  # last 100 scores
# isTest=True -> perform grasping on test set of objects
# env = jacoDiverseObjectEnv(renders=True, isDiscrete=True, removeAutoXDistance=False, maxSteps=20, isTest=True)
env = jacoDiverseObjectEnv(actionRepeat=80, renders=True, isDiscrete=True, maxSteps=30, dv=0.02,
                           removeAutoXDistance=False, width=64, height=64, isTest=True)
env.reset()

init_screen = get_screen(environment=env)


_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n  # Get number of actions from gym action space
policy_net = DQN(screen_height, screen_width, n_actions).to(device)

# load trained model for the policy network
checkpoint = torch.load(PATH, map_location=device)
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

s=0
f=0
reward = 0
# evaluate the trained model by selecting actions according to the policy net
for i_episode in range(number_episodes):
    env.reset()
    state = get_screen(environment=env)

    p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "video/final/clips/"+str(i_episode)+".mp4")
    stacked_states = collections.deque(STACK_SIZE*[state], maxlen=STACK_SIZE)
    for t in count():
        stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
        # Select and perform an action
        action = policy_net(stacked_states_t).max(1)[1].view(1, 1)
        _, reward, done, _ = env.step(action.item())
        # Observe new state
        next_state = get_screen(environment=env)
        stacked_states.append(next_state)
        if done:
            break


