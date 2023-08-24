"""
Baseline DQN code from: 
    https://github.com/mahyaret/kuka_rl/blob/master/kuka_rl.ipynb
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    https://source.coderefinery.org/JosteinDanielsen/msc-thesis/-/tree/master/kuka_eye_dqn

"""

import random
import numpy as np
from collections import namedtuple
import collections
from itertools import count
import timeit
from datetime import timedelta
from PIL import Image
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


from jaco_env_extended import jacoDiverseObjectEnv
import pybullet as pb

# Set seed fixed for reproducibility, also try different seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Load env
env = jacoDiverseObjectEnv(actionRepeat=80, renders=False, isDiscrete=True, maxSteps=30, dv=0.02,
                           removeAutoXDistance=False, width=64, height=64)
env.cid = pb.connect(pb.DIRECT)

# Choose system (CPU/GPU), depending if Nvidia Cuda is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Define replay buffer
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves the experience."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity # Dynamic replay memory -> delete oldest entry first and add newest if buffer full

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



'''
DQN
'''

# Define deep q network
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

'''
Input extraction
'''
# Preprocess image
preprocess = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize(128, interpolation=Image.CUBIC),
                    T.ToTensor()])

# Function for recieving PyBullet camera data as input image
def get_screen():
    global stacked_screens
    # Transpose screen into torch order (CHW).
    rgb, depth, segmentation = env._get_observation()
    screen = rgb.transpose((2, 0, 1))   #[rgb.transpose((2, 0, 1)),depth.transpose((2, 0, 1)),segmentation] 

    # Convert to float, rescale, convert to torch tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return preprocess(screen).unsqueeze(0).to(device)

# Select action (random action/best action) based on epsiolon greedy strategy
def select_action(state, i_episode):
    global steps_done
    global eps_threshold
    sample = random.random()
    # Linear decaying epsilon, also try exponential etc.
    eps_threshold = max(EPS_END, EPS_START - i_episode / EPS_DECAY_LAST_FRAME)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

'''
Training loop
'''

# Update network
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


import datetime
def log(m):
    ct = datetime.datetime.now()
    ts = ct.timestamp()
    mess = "" + str(ct) + " - " + str(m)
    LOGFILE_POINTER.write(mess + "\n")
    LOGFILE_POINTER.flush()
    if LOG_ON_SCREEN:
      print(mess)
    return ts

'''
Main training loop
'''

PATH = 'policy_dqn.pt'
# Declare number of total episodes
num_episodes = 50000        #[25000,100000,500000]               
writer = SummaryWriter()
total_rewards = []
ten_rewards = 0
best_mean_reward = None
start_time = timeit.default_timer()
logfilename = 'learnDQN.log'

env.reset()


'''
Training
*Instantiate DQN.
*Epsilon greedy action selection with epsilon decay:
probability of choosing a random action will start at EPS_START and 
will decay exponentially towards EPS_END. EPS_DECAY controls the rate of the decay.
'''

# Hyperparameters, search for different combinations
BATCH_SIZE = 64 
GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.1
EPS_DECAY_LAST_FRAME = 2*10**4
TARGET_UPDATE = 1000
LEARNING_RATE = 1e-4
REPLAY_BUFFER_SIZE = 30000
Stack_Size = 4

# Get screen size so that we can initialize layers correctly based on shape
# returned from pybullet (128, 128, 3).
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

eps_threshold = 0
STACK_SIZE = 0
memory = None
policy_net = None
target_net = None

LOGFILE=""
LOGFILE_POINTER = None
LOG_ON_SCREEN = False


if __name__ == "__main__":
  import datetime
  import optparse
  ts_start = datetime.datetime.now()
  parser = optparse.OptionParser()

  parser.add_option('-l', '--logfile',
                    action="store", 
                    dest="logfile_name",
                    help="Name of logfile", 
                    default="")
  parser.add_option('-d', '--detail_level',
                    action="store", 
                    dest="detail_level",
                    help="Level of detail (a,e,p)", 
                    default="a")
  parser.add_option('-a', '--logging on both screen and logfile (default)',
                    action="store", 
                    dest="both",
                    help="Logging on screen and logfile", 
                    default=True)
  
  options, args = parser.parse_args()
  
  if options.both == True:
    LOG_ON_SCREEN = True
  else:
    LOG_ON_SCREEN = False

  # Init replay buffer, policy net and target net
  STACK_SIZE = Stack_Size
  memory = ReplayMemory(REPLAY_BUFFER_SIZE)
  policy_net = DQN(screen_height, screen_width, n_actions).to(device)
  target_net = DQN(screen_height, screen_width, n_actions).to(device)
  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval()

  # Use ADAM as optimizer
  optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
  
  # Name of saved .log file and .pt model
  filename = "bs"+str(BATCH_SIZE)+"_ss" + str(STACK_SIZE) + "_rb" + str(REPLAY_BUFFER_SIZE)+"_gamma"+str(GAMMA)+"_decaylf"+str(EPS_DECAY_LAST_FRAME)+"_lr"+str(LEARNING_RATE)

  if options.logfile_name == "":
    LOGFILE = "logs/learnDQN_phase2_"+ filename + ".log"
    print("Logging progress to: " + LOGFILE)
    print("Detail level of logging is set to \'" + options.detail_level + "' - ", end = "")
    if (options.detail_level == 'p'):
      print("Print progress in terms of increased mean rewards")
    elif (options.detail_level == 'e'):
      print("Print progress in terms of timestamped Epoc's ")
    else:
      print("Print all logging info")

  # Path of saved model
  PATH = "models/policyDQN_phase2_" + filename + ".pt" 
  if LOGFILE_POINTER == None:
    LOGFILE_POINTER = open(LOGFILE,"w")

  old_reward_ts = log("Starting learning loop ...\n==================================================================")
  log("Using Framebuffer of: " + str(STACK_SIZE) + " frames\t and \tReplayBufferSize: " +  str(REPLAY_BUFFER_SIZE))
  log("Loglevel is set to: " + str(options.detail_level))
  
  old_epoch_ts = -1
  # Begin training
  for i_episode in range(num_episodes):
      # Initialize the environment and state
      if i_episode % 10 == 0:
          ct = datetime.datetime.now()
          new_epoch_ts = ct.timestamp()
          if old_epoch_ts > 0:
              diff = new_epoch_ts - old_epoch_ts
          else:
              diff = ""
          if options.detail_level != 'p':
            old_epoch_ts = log("Epoc #\t" + str(i_episode) + "\t " + str(diff))
      env.reset()
      state = get_screen()
      stacked_states = collections.deque(STACK_SIZE*[state],maxlen=STACK_SIZE)
      for t in count():
          stacked_states_t =  torch.cat(tuple(stacked_states),dim=1)
          # Select and perform an action
          action = select_action(stacked_states_t, i_episode)
          _, reward, done, _ = env.step(action.item())
          reward = torch.tensor([reward], device=device)
  
          # Observe new state
          next_state = get_screen()
          if not done:
              next_stacked_states = stacked_states
              next_stacked_states.append(next_state)
              next_stacked_states_t =  torch.cat(tuple(next_stacked_states),dim=1)
          else:
              next_stacked_states_t = None
  
          # Store the transition in memory
          memory.push(stacked_states_t, action, next_stacked_states_t, reward)
  
          # Move to the next state
          stacked_states = next_stacked_states
  
          # Perform one step of the optimization (on the target network)
          optimize_model()
          if done:
              reward = reward.cpu().numpy().item()
              ten_rewards += reward
              total_rewards.append(reward)
              mean_reward = np.mean(total_rewards[-100:])*100
              writer.add_scalar("epsilon", eps_threshold, i_episode)
              if (best_mean_reward is None or best_mean_reward < mean_reward) and i_episode > 100:
                  # For saving the model and possibly resuming training
                  torch.save({
                          'policy_net_state_dict': policy_net.state_dict(),
                          'target_net_state_dict': target_net.state_dict(),
                          'optimizer_policy_net_state_dict': optimizer.state_dict()
                          }, PATH)
                  if best_mean_reward is not None:
                      print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                      ct = datetime.datetime.now()
                      new_reward_ts = ct.timestamp()
                      diff = new_reward_ts - old_reward_ts
                      if (options.detail_level != 'e'):
                        old_rewards_ts = log("Time between reward step: #\t" + str(diff))
                        s = 'Average Score: {:.3f}'.format(mean_reward)
                        elapsed = timeit.default_timer() - start_time
                        t = "Elapsed time: {}".format(timedelta(seconds=elapsed))
                        log("" + s + " " + t)
  
                  best_mean_reward = mean_reward
              break
  
      if i_episode%10 == 0:
              writer.add_scalar('ten episodes average rewards', ten_rewards/10.0, i_episode)
              ten_rewards = 0
  
      # Update the target network, copying all weights and biases in DQN
      if i_episode % TARGET_UPDATE == 0:
          target_net.load_state_dict(policy_net.state_dict())
      # Declare termination condition of the training
      if i_episode>=200 and mean_reward>90:
          print('Environment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode+1, mean_reward))
          break
  
  
  print('Average Score: {:.3f}'.format(mean_reward))
  elapsed = timeit.default_timer() - start_time
  print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
  writer.close()
  env.close()

'''
Evaluation
'''
episode = 10
scores_window = collections.deque(maxlen=100)  # Last 100 scores
env = jacoDiverseObjectEnv(actionRepeat=80, renders=False, isDiscrete=True, maxSteps=30, dv=0.02,
                           removeAutoXDistance=False, width=64, height=64, isTest=True)
env.cid = pb.connect(pb.DIRECT)
# Load the model
checkpoint = torch.load(PATH)
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

# Evaluate the model
for i_episode in range(episode):
    env.reset()
    state = get_screen()
    stacked_states = collections.deque(STACK_SIZE*[state],maxlen=STACK_SIZE)
    for t in count():
        stacked_states_t =  torch.cat(tuple(stacked_states),dim=1)
        # Select and perform an action
        action = policy_net(stacked_states_t).max(1)[1].view(1, 1)
        _, reward, done, _ = env.step(action.item())
        # Observe new state
        next_state = get_screen()
        stacked_states.append(next_state)
        if done:
            break
    print("Episode: {0:d}, reward: {1}".format(i_episode+1, reward), end="\n")