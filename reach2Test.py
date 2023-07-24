import os
from gym import utils
import spacerobot_envtest

MODEL_XML_PATH = os.path.join('mujoco_files','spacerobot','spacerobot_v3.xml')

class SpaceReachEnv(spacerobot_envtest.SpacerobotEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'arm:shoulder_pan_joint': 0,
            'arm:shoulder_lift_joint': 0,
            'arm:elbow_joint': 0.0,
            'arm:wrist_1_joint': 0.0,
            'arm:wrist_2_joint': 0.0,
            'arm:wrist_3_joint': 0.0
        }
        spacerobot_envtest.SpacerobotEnv.__init__(
            self, MODEL_XML_PATH, n_substeps=20, distance_threshold=0.15,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
