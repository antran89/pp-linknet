'''
Author: an.tran@grab.com
Predicting the road mask in the satellite image.
'''
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import argparse
from time import time
import logging
import copy

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool, DinkNet34_abn, DinkNet34_context_abn, DinkNet34_psp, DinkNet34_psp64, DinkNet34_psp64_hdc, DinkNet34_abn_psp128, DinkNet34_logits
import apollo_python_common.io_stream.io_utils as io_utils
from apollo_python_common.ml_pipeline.multi_threaded_predictor import MultiThreadedPredictor
#from missing_roads.scripts.common.io.es_interface import ES_Interface
from apollo_python_common.proto_utils.proto_api import MQ_Messsage_Type
from satellite_missing_roads.scripts.pp_linknet.road_mask_predictor import RoadMaskPredictor
os.environ["NUMEXPR_MAX_THREADS"] = "20"


class RoadMaskPredictorMQ(MultiThreadedPredictor):
    '''Predict road masks.
    '''

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        #self.es = ES_Interface(config)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        #logging.basicConfig(filename=self.config['log_file'], level=logging.INFO)

    def preprocess(self, sat_image_proto):
        print("------INPUT-------")
        self.logger.info(f"Predicting road masks for city: {sat_image_proto.city_id}")
        print(sat_image_proto)
        
        # combining local config with sat_image_proto
        config = copy.deepcopy(self.config)
        config['city_id'] = sat_image_proto.city_id
        config['dataset_root_path'] = sat_image_proto.dataset_root_path
        builder = RoadMaskPredictor(config)
        return (builder)
    
    def predict(self, builders):
        for builder in builders:
            builder.generate_dataset()
        return [None]
        
    def postprocess(self, msg, sat_image_proto):
        print(f"Finished predicting road masks of city: {sat_image_proto.city_id}")
        self.logger.info(f"Finished predicting road masks of city: {sat_image_proto.city_id}")
        print(sat_image_proto)
        
        return sat_image_proto


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file",
                        help="config file path", type=str, required=True)
    return parser.parse_args()


def run_dataset_builder(conf_file: str) -> None:
    conf = dict(io_utils.config_load(conf_file))
    RoadMaskPredictorMQ(conf, mq_message_type=MQ_Messsage_Type.SAT_DATA).start()


if __name__ == '__main__':
    args = parse_arguments()
    run_dataset_builder(args.config_file)
