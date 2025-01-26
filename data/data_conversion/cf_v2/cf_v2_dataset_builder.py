from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import pickle
import os
from PIL import Image

RESIZE = (128, 128)

class CfDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(RESIZE[0], RESIZE[1], 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(3,),
                            dtype=np.float64,
                            doc='Robot state, consists of [2x position, 1x yaw]',
                        ),
                        'position': tfds.features.Tensor(
                            shape=(2,),
                            dtype=np.float64,
                            doc='Robot position',
                        ),
                        'yaw': tfds.features.Tensor(
                            shape=(1,),
                            dtype=np.float64,
                            doc='Robot yaw',
                        ),
                        'yaw_rotmat': tfds.features.Tensor(
                            shape=(3, 3),
                            dtype=np.float64,
                            doc='Robot yaw rotation matrix',
                        ),

                    }),
                    'action': tfds.features.Tensor(
                        shape=(2,),
                        dtype=np.float64,
                        doc='Robot action, consists of 2x position'
                    ),
                     'action_angle': tfds.features.Tensor(
                        shape=(3,),
                        dtype=np.float64,
                        doc='Robot action, consists of 2x position, 1x yaw',
                    ),

                    'discount': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float64,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                   'language_instruction': tfds.features.Tensor(
                        shape=(10,),
                        dtype=tf.string,
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'episode_id': tfds.features.Scalar(
                        dtype=tf.int32,
                        doc='Episode ID.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/hdd/cf_v2_dataset/train'),
            'val': self._generate_examples(path='/hdd/cf_v2_dataset/val'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        id_iter = 0
        """Generator of examples for each split."""
        def _get_folder_names(data_dir):
            folder_names = [
                f for f in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, f))
                and "traj_data_filtered.pkl" in os.listdir(os.path.join(data_dir, f))
            ]   
            folder_names = [os.path.join(data_dir, f) for f in folder_names]

            return folder_names

        def _yaw_rotmat(yaw: float) -> np.ndarray:
            return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],)


        def _to_local_coords(
                positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
            ) -> np.ndarray:
            """
            Convert positions to local coordinates

            Args:
                positions (np.ndarray): positions to convert
                curr_pos (np.ndarray): current position
                curr_yaw (float): current yaw
            Returns:
                np.ndarray: positions in local coordinates
            """
            rotmat = _yaw_rotmat(curr_yaw)
            if positions.shape[-1] == 2:
                rotmat = rotmat[:2, :2]
            elif positions.shape[-1] == 3:
                pass
            else:
                raise ValueError

            return (positions - curr_pos).dot(rotmat)


        def _process_image(path, mode='stretch'):
            try:
                img = Image.open(path)
            except:
                img = Image.new('RGB', RESIZE, (255, 255, 255))
            if mode == 'stretch':
                img = img.resize(RESIZE)
            elif mode == 'crop':
                img = img.resize((85, 64))
                
                top = 0
                bottom = 64
                left = (85 - 64) // 2
                right = (85 + 64) // 2
                img = img.crop((left, top, right, bottom))
            
            return np.asarray(img, dtype='uint8')
        
        def _parse_language(language_instruction):
            if type(language_instruction) == list:
                language_instruction = language_instruction[0]
            elif type(language_instruction) == str:
                pass
            elif type(language_instruction) == dict:
                if "instruction" in language_instruction.keys():
                    language_instruction = language_instruction["instruction"]
                elif "type" in language_instruction.keys() and "landmark" in language_instruction.keys():
                    language_instruction = f"{language_instruction['type']} {language_instruction['landmark']}"
                elif "type" in language_instruction.keys() and "manner" in language_instruction.keys():
                    language_instruction = f"{language_instruction['type']} {language_instruction['manner']}"
                elif "type" in language_instruction.keys() and "direction" in language_instruction.keys():
                    language_instruction = f"{language_instruction['type']} {language_instruction['direction']}"
                elif "type" in language_instruction.keys() and "description" in language_instruction.keys():
                    language_instruction = f"{language_instruction['type']} {language_instruction['description']}"
                else:
                    return None
            return language_instruction

        def _compute_actions(traj_data, curr_time, goal_time, len_traj_pred=1, waypoint_spacing=1, learn_angle=True, normalize=False):
            start_index = curr_time
            end_index = curr_time + len_traj_pred * waypoint_spacing + 1
            yaw = traj_data["yaw"][start_index:end_index:waypoint_spacing]
            positions = traj_data["position"][start_index:end_index:waypoint_spacing]
            goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

            if len(yaw.shape) == 2:
                yaw = yaw.squeeze(1)

            if yaw.shape != (len_traj_pred + 1,):
                const_len = len_traj_pred + 1 - yaw.shape[0]
                yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
                positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)
            yaw = yaw[:len_traj_pred + 1]
            positions = positions[:len_traj_pred + 1]
            assert yaw.shape == (len_traj_pred + 1,), f"{yaw.shape} and {(len_traj_pred + 1,)} should be equal"
            assert positions.shape == (len_traj_pred + 1, 2), f"{positions.shape} and {(len_traj_pred + 1, 2)} should be equal"

            waypoints = _to_local_coords(positions, positions[0], yaw[0])
            goal_pos = _to_local_coords(goal_pos, positions[0], yaw[0])

            assert waypoints.shape == (len_traj_pred + 1, 2), f"{waypoints.shape} and {(len_traj_pred + 1, 2)} should be equal"

            if learn_angle:
                yaw = yaw[1:] - yaw[0]
                actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
            else:
                actions = waypoints[1:]

            if normalize:
                actions[:, :2] /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
                goal_pos /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing

            return actions, goal_pos

        def _parse_example(episode):
            episode_path, episode_id = episode
            # load raw data --> this should change for your dataset
            data_path = os.path.join(episode_path, 'traj_data_filtered.pkl')
            data = np.load(data_path, allow_pickle=True)     # this is a list of dicts in our case
            if type(data["yaw"]) == list:
                data["yaw"] = np.array(data["yaw"])
            if data["position"].shape[0] < data["yaw"].shape[0]:
                data["yaw"] = data["yaw"][:data["position"].shape[0]]
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            samples = []
            successes = []
            episode_paths = []
            episode = []
            for i in range(len(data['position'])):
                # compute Kona language embedding
                language_instructions = [_parse_language(lang["traj_description"]) for lang in data['language_annotations']]
                language_instructions = [lang for lang in language_instructions if lang is not None]
                
                if len(language_instructions) > 10:
                    language_instructions = language_instructions[:10]
                else:
                    language_instructions += [''] * (10 - len(language_instructions))

                #Get image observation
                image_path = f'{i}.jpg'
                img = _process_image(os.path.join(episode_path, image_path), mode='stretch')

                #Get state observation
                position = data['position'][i]
                yaw = data['yaw'][i].reshape(-1)
                state = np.concatenate((position, yaw))
                yaw_rotmat = _yaw_rotmat(yaw[0])

                # Recover action(s)
                action, goal_pos = _compute_actions(data, i, i+1, len_traj_pred=1,
                    waypoint_spacing=1, learn_angle=False, normalize=False) 
                action_angle, goal_pos = _compute_actions(data, i, i+1, len_traj_pred=1,
                    waypoint_spacing=1, learn_angle=True, normalize=False)
                action = action[0]
                action_angle = action_angle[0]

                episode.append({
                    'observation': {
                        'image': img,
                        'state': state,
                        'position': position,
                        'yaw': yaw,
                        'yaw_rotmat': yaw_rotmat,
                    },
                    'action': action,
                    'action_angle': action_angle,
                    'discount': 1.0,
                    'reward': float(i == (len(data['position']) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data['position']) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': language_instructions,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path,
                    'episode_id': episode_id,
                }
            }

            success = len(data['position']) >= 1

            if success:
                return episode_path, sample
            else:
                return None
            

        dataset_names = os.listdir(path)
        episode_paths = []
        for name in dataset_names:
            episode_paths += _get_folder_names(os.path.join(path, name))
        episode_ids = np.arange(len(episode_paths))
        episodes = [(episode_path, episode_id) for episode_path, episode_id in zip(episode_paths, episode_ids)]

        # for smallish datasets, use single-thread parsing
        for sample, sample_id in zip(episode_paths, episode_ids):
            yield _parse_example((sample, sample_id))

        # # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episodes)
        #         | beam.Map(_parse_example)
        # )

