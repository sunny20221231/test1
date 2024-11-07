import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# 设置日志级别是3
from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

