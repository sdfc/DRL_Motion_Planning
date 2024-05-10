import inspect
import os

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(current_dir)
# sys.path.append('../../')

from cr5_env.cr5_gripper_obstacle_visual_mlp import CR5AvoidVisualEnv
from ppo.ppo_cuda import ppo
from spinup.utils.mpi_tools import mpi_fork
import ppo.core_cuda as core

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--is_render', action="store_true", default=False)
parser.add_argument('--is_good_view', action="store_true", default=False)

parser.add_argument('--hid', type=int, default=64)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--mpi_num', type=int, default=1)
parser.add_argument('--max_steps', type=int, default=5000)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--exp_name', type=str, default='ppo-gripper-avoid-without-ou-fixedAll')
parser.add_argument('--log_dir', type=str, default="/home/magic/ZhangGZ/intention-aware-HRC/drl_motion_planning/logs")
args = parser.parse_args()

env = CR5AvoidVisualEnv(is_render=args.is_render, is_good_view=args.is_good_view, max_steps=args.max_steps)

mpi_fork(args.mpi_num)  # run parallel code with mpi

from spinup.utils.run_utils import setup_logger_kwargs

logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=args.log_dir, datestamp=True)

ppo(env,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
    gamma=args.gamma,
    seed=args.seed,
    steps_per_epoch=args.max_steps * args.mpi_num,
    epochs=args.epochs,
    logger_kwargs=logger_kwargs)
