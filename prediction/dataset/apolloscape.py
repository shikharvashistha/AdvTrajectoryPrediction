import os
import numpy as np

from .base import BaseDataset


class ApolloscapeDataset(BaseDataset):
    def __init__(self, obs_length, pred_length, time_step=0.5):
        super().__init__(obs_length, pred_length, time_step)

        self.data_dir =  os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/apolloscape/")
        self.test_data_dir = os.path.join(self.data_dir, "prediction_test")
        self.val_data_dir = os.path.join(self.data_dir, "prediction_val")
        self.train_data_dir = os.path.join(self.data_dir, "prediction_train")
        self.default_time_step = 0.5
        self.skip_step = int(self.time_step / self.default_time_step)

    def format_data_generator(self, data_dir, batch_size=64, enable_batch=True, allow_incomplete_traces=True):
        files = os.listdir(data_dir)
        for filename in files:
            file_path = os.path.join(data_dir, filename)
            data = np.genfromtxt(file_path, delimiter=" ")
            data = data[~(data[:, 2] == 5)]
            start_frame_id = int(np.min(data[:,0]))

            numFrames = len(np.unique(data[:, 0]))
            numSlices = numFrames - (self.seq_length - 1) * self.skip_step + 1 + 1

            if enable_batch:
                batch = []

            for slice_id in range(numSlices):
                input_data = {
                    "observe_length": self.obs_length,
                    "predict_length": self.pred_length,
                    "time_step": self.time_step,
                    "objects": {}
                }

                # fill data
                for local_frame_id in range(self.seq_length):
                    frame_id = start_frame_id + slice_id + local_frame_id * self.skip_step
                    frame_data = data[data[:, 0] == frame_id, :]

                    for obj_index in range(frame_data.shape[0]):
                        obj_data = frame_data[obj_index, :]
                        obj_id = str(int(obj_data[1]))

                        if obj_id not in input_data["objects"]:
                            if local_frame_id < self.obs_length:
                                input_data["objects"][obj_id] = {
                                    "type": int(obj_data[2]),
                                    "complete": True,
                                    "observe_trace": np.zeros((self.obs_length,2)),
                                    "observe_feature": np.zeros((self.obs_length,5)),
                                }
                                if self.pred_length > 0:
                                    input_data["objects"][obj_id]["future_trace"] = np.zeros((self.pred_length,2))
                                    input_data["objects"][obj_id]["future_feature"] = np.zeros((self.pred_length,5))
                                    input_data["objects"][obj_id]["predict_trace"] = np.zeros((self.pred_length,2))
                            else:
                                continue

                        obj = input_data["objects"][obj_id]
                        if local_frame_id < self.obs_length:
                            obj["observe_trace"][local_frame_id, :] = obj_data[3:5]
                            obj["observe_feature"][local_frame_id, :] = obj_data[5:]
                        else:
                            obj["future_trace"][local_frame_id-self.obs_length, :] = obj_data[3:5]
                            obj["future_feature"][local_frame_id-self.obs_length, :] = obj_data[5:]

                # remove invalid data
                invalid_obj_ids = []
                for obj_id, obj in input_data["objects"].items():
                    if np.sum(obj["observe_trace"]) <= 0.0001:
                        invalid_obj_ids.append(obj_id)
                    if np.min(np.concatenate((obj["observe_trace"], obj["predict_trace"]), axis=0)) <= 0.0001:
                        if not allow_incomplete_traces:
                            invalid_obj_ids.append(obj_id)
                        else:
                            obj["complete"] = False
                for invalid_obj_id in invalid_obj_ids:
                    del input_data["objects"][invalid_obj_id]
                
                # may create empty data, especially after invalid data removal
                if len(input_data["objects"]) == 0:
                    continue

                if enable_batch:
                    batch.append(input_data)

                    if len(batch) == batch_size:
                        yield batch
                        batch = []
                else:
                    yield input_data

    def train_data_generator(self, **kwargs):
        return self.format_data_generator(self.train_data_dir, **kwargs)

    def val_data_generator(self, **kwargs):
        return self.format_data_generator(self.val_data_dir, **kwargs)

    def test_data_generator(self, **kwargs):
        return self.format_data_generator(self.test_data_dir, **kwargs)