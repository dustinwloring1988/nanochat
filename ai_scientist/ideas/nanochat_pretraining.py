import os
import sys

working_dir = os.path.join(os.getcwd(), "working")
source_dir = os.environ["NANOCHAT_SOURCE_DIR"]
sys.path.insert(0, source_dir)

from nanochat.ai_scientist_experiment import run_from_environment

run_from_environment(working_dir)
