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
import geohash
import pandas as pd

from networks.unet import Unet
from networks.dunet import Dunet
from networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool, DinkNet34_abn, DinkNet34_context_abn, DinkNet34_psp, DinkNet34_psp64, DinkNet34_psp64_hdc, DinkNet34_abn_psp128, DinkNet34_logits
import apollo_python_common.io_stream.io_utils as io_utils
from apollo_python_common.ml_pipeline.multi_threaded_predictor import MultiThreadedPredictor
#from missing_roads.scripts.common.io.es_interface import ES_Interface
from apollo_python_common.proto_utils.proto_api import MQ_Messsage_Type
from satellite_missing_roads.scripts.pp_linknet.road_mask_predictor import RoadMaskPredictor
from satellite_missing_roads.scripts.pp_linknet.road_mask_predictor_azure import RoadMaskPredictorAzure
from satellite_imagery_funcs import get_lon_lat_from_image_name
from io_interface import IO_Interface
from constants import Constants as c
os.environ["NUMEXPR_MAX_THREADS"] = "20"


class RoadMaskPredictorGeohashMQ(MultiThreadedPredictor):
    '''Predict road masks.
    '''

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        #self.es = ES_Interface(config)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        #logging.basicConfig(filename=self.config['log_file'], level=logging.INFO)
        self.io = IO_Interface(config)
    
    def init_config(self, config: dict) -> dict:
        """ Initilize the configuration and create folders
        """
        if config['city_id'] == '':
            config['city_id'] = config['geohash']
        if config['city_id'][-1] == '/':
            config['city_id'] = config['city_id'][:-1]
        config['dataset_root_path'] = os.path.join(config['dataset_root_path'], 'gsp')
        config['geohash_assignment_file'] = os.path.join(config['dataset_root_path'], config['city_id'], '%s_gh5.csv' % config['city_id'])
        jpg_out_folder = os.path.join(config['dataset_root_path'], config['city_id'], 'overlapping-jpg')
        config['jpg_out_folder'] = jpg_out_folder
        transformation_file = os.path.join(config['dataset_root_path'], config['city_id'], '%s.pkl' % config['city_id'])
        config['transformation_file'] = transformation_file
        config['geohash_assignment_file'] = os.path.join(config['dataset_root_path'], config['city_id'], '%s_gh5.csv' % config['city_id'])
        config['geohash_assignment_file_all'] = os.path.join(config['dataset_root_path'], config['city_id'], '%s_gh5_all.csv' % config['city_id'])
        return config

    def get_image_geohash_assignment(self, config, geohash_lvl=5):
        ''' getting assignment of geohash on all images in jpg_out_folder folder, and save it into a csv file.
        '''
        crop_size = 1024
        # img_files = glob.glob(os.path.join(config['jpg_out_folder'], '*.jpg'))
        # img_files.sort()
        img_files = self.io.get_files_from_folder(config['jpg_out_folder'])
        img_files = [p for p in img_files if os.path.splitext(p)[1] == '.jpg']
        transformation_df = self.io.get_pickle_df(config['transformation_file'])
        # getting lon/lat of the center point of the image
        d = {'JPGImageName': [], 'GeohashAssignment': []}
        d_all = {'JPGImageName': [], 'GeohashAssignment': []}
        for img_path in img_files:
            img_name = os.path.basename(img_path)
            trunk = img_name.split('_')
            r = int(trunk[-3])
            c = int(trunk[-2])
            # center point image name
            r1 = r + crop_size / 2
            c1 = c + crop_size / 2
            trunk[-3] = str(int(r1))
            trunk[-2] = str(int(c1))
            center_img_name = '_'.join(trunk)
            lon, lat = get_lon_lat_from_image_name(center_img_name, transformation_df, geohash_lvl)
            assignment_gh = geohash.encode(latitude=lat, longitude=lon, precision=geohash_lvl)
            d_all['JPGImageName'].append(img_name)
            d_all['GeohashAssignment'].append(assignment_gh)
            if r % crop_size == 0 and c % crop_size == 0:
                d['JPGImageName'].append(img_name)
                d['GeohashAssignment'].append(assignment_gh)
        df = pd.DataFrame(data=d)
        self.io.save_csv_df_to_disk(df, config['geohash_assignment_file'])
        df_all = pd.DataFrame(data=d_all)
        self.io.save_csv_df_to_disk(df_all, config['geohash_assignment_file_all'])

    def preprocess(self, sat_image_proto):
        print("------INPUT-------")
        self.logger.info(f"Predicting road masks for city: {sat_image_proto.city_id}")
        print(sat_image_proto)

        # combining local config with sat_image_proto
        config = copy.deepcopy(self.config)
        config['city_id'] = sat_image_proto.city_id
        config['dataset_root_path'] = sat_image_proto.dataset_root_path
        config['geohash'] = sat_image_proto.geohash
        config = self.init_config(config)

        # saving geohash level 5 labels of all jpg images
        # temporarily don't need to get the geohash_assignment_file
        # if not self.io.file_exists(config['geohash_assignment_file']):
        #     if config['geohash']:
        #         self.get_image_geohash_assignment(config, geohash_lvl=len(config['geohash']))
        #     else:
        #         self.get_image_geohash_assignment(config, geohash_lvl=5)

        if self.config[c.STORAGE_TYPE] == c.AZURE_STORAGE_TYPE:
            builder = RoadMaskPredictorAzure(config)
        else:
            builder = RoadMaskPredictor(config)
        return (builder)
    
    def predict(self, builders):
        for builder in builders:
            builder.generate_dataset()
        return [None]
        
    def postprocess(self, msg, sat_image_proto):
        print(f"Finished predicting road masks for geohash {sat_image_proto.geohash} of city: {sat_image_proto.city_id}")
        self.logger.info(f"Finished predicting road masks for geohash {sat_image_proto.geohash} of city: {sat_image_proto.city_id}")
        print(sat_image_proto)
        
        return sat_image_proto


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file",
                        help="config file path", type=str, required=True)
    return parser.parse_args()


def run_dataset_builder(conf_file: str) -> None:
    conf = dict(io_utils.config_load(conf_file))
    RoadMaskPredictorGeohashMQ(conf, mq_message_type=MQ_Messsage_Type.SAT_DATA).start()


if __name__ == '__main__':
    args = parse_arguments()
    run_dataset_builder(args.config_file)
