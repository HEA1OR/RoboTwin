import sys

import os
import h5py
import numpy as np
import pickle
import cv2
import argparse
import yaml, json


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        left_gripper, left_arm = (
            root["/joint_action/left_gripper"][()],
            root["/joint_action/left_arm"][()],
        )
        right_gripper, right_arm = (
            root["/joint_action/right_gripper"][()],
            root["/joint_action/right_arm"][()],
        )
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]
        observer_arm = None
        observer_in_operator_spherical = None
        observer_dir_offset_angles = None
        if "/frame_tracking/observer_arm" in root:
            observer_arm = root["/frame_tracking/observer_arm"][()]
        if "/frame_tracking/observer_in_operator_spherical" in root:
            observer_in_operator_spherical = root["/frame_tracking/observer_in_operator_spherical"][()]
        if "/frame_tracking/observer_dir_offset_angles" in root:
            observer_dir_offset_angles = root["/frame_tracking/observer_dir_offset_angles"][()]

    return (
        left_gripper,
        left_arm,
        right_gripper,
        right_arm,
        image_dict,
        observer_arm,
        observer_in_operator_spherical,
        observer_dir_offset_angles,
    )


def parse_observer_arm(raw_value):
    if isinstance(raw_value, bytes):
        arm_str = raw_value.decode("utf-8", errors="ignore").strip("\x00").strip().lower()
    elif isinstance(raw_value, np.bytes_):
        arm_str = raw_value.tobytes().decode("utf-8", errors="ignore").strip("\x00").strip().lower()
    else:
        arm_str = str(raw_value).strip().lower()

    # state coding: operate=0, observe=1
    if arm_str.startswith("left"):
        return 1.0, 0.0
    if arm_str.startswith("right"):
        return 0.0, 1.0
    return 0.0, 0.0


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        success, encoded_image = cv2.imencode(".jpg", imgs[i])
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b"\0"))
    return encode_data, max_len


def get_task_config(task_name):
    with open(f"./task_config/{task_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return args


def data_transform(path, episode_num, save_path):
    begin = 0
    floders = os.listdir(path)
    # assert episode_num <= len(floders), "data num not enough"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(episode_num):

        desc_type = "seen"
        instruction_data_path = os.path.join(path, "instructions", f"episode{i}.json")
        with open(instruction_data_path, "r") as f_instr:
            instruction_dict = json.load(f_instr)
        instructions = instruction_dict[desc_type]
        save_instructions_json = {"instructions": instructions}

        os.makedirs(os.path.join(save_path, f"episode_{i}"), exist_ok=True)

        with open(
                os.path.join(os.path.join(save_path, f"episode_{i}"), "instructions.json"),
                "w",
        ) as f:
            json.dump(save_instructions_json, f, indent=2)

        (
            left_gripper_all,
            left_arm_all,
            right_gripper_all,
            right_arm_all,
            image_dict,
            observer_arm_all,
            observer_in_operator_spherical_all,
            observer_dir_offset_angles_all,
        ) = load_hdf5(os.path.join(path, "data_with_observer", f"episode{i}.hdf5"))
        qpos = []
        actions = []
        cam_high = []
        cam_right_wrist = []
        cam_left_wrist = []
        left_arm_dim = []
        right_arm_dim = []
        left_arm_state = []
        right_arm_state = []
        perception_arm_dim = []
        perception_qpos = []

        for j in range(0, left_gripper_all.shape[0]):

            left_gripper, left_arm, right_gripper, right_arm = (
                left_gripper_all[j],
                left_arm_all[j],
                right_gripper_all[j],
                right_arm_all[j],
            )

            state = np.array(left_arm.tolist() + [left_gripper] + right_arm.tolist() + [right_gripper])  # joints angle

            state = state.astype(np.float32)

            if j != left_gripper_all.shape[0] - 1:
                qpos.append(state)
                if observer_arm_all is None:
                    l_state, r_state = 0.0, 0.0
                else:
                    l_state, r_state = parse_observer_arm(observer_arm_all[j])
                left_arm_state.append(l_state)
                right_arm_state.append(r_state)

                if (
                    observer_in_operator_spherical_all is not None
                    and observer_dir_offset_angles_all is not None
                ):
                    p_qpos = np.concatenate(
                        [observer_in_operator_spherical_all[j], observer_dir_offset_angles_all[j]],
                        axis=0,
                    )
                else:
                    p_qpos = np.zeros((5,), dtype=np.float32)
                perception_qpos.append(p_qpos.astype(np.float32))
                # perception_arm_dim records the feature dimension of perception_qpos.
                perception_arm_dim.append(float(p_qpos.shape[0]))
                left_arm_dim.append(float(left_arm.shape[0]))
                right_arm_dim.append(float(right_arm.shape[0]))

                camera_high_bits = image_dict["head_camera"][j]
                camera_high = cv2.imdecode(np.frombuffer(camera_high_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_high_resized = cv2.resize(camera_high, (640, 480))
                cam_high.append(camera_high_resized)

                camera_right_wrist_bits = image_dict["right_camera"][j]
                camera_right_wrist = cv2.imdecode(np.frombuffer(camera_right_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_right_wrist_resized = cv2.resize(camera_right_wrist, (640, 480))
                cam_right_wrist.append(camera_right_wrist_resized)

                camera_left_wrist_bits = image_dict["left_camera"][j]
                camera_left_wrist = cv2.imdecode(np.frombuffer(camera_left_wrist_bits, np.uint8), cv2.IMREAD_COLOR)
                camera_left_wrist_resized = cv2.resize(camera_left_wrist, (640, 480))
                cam_left_wrist.append(camera_left_wrist_resized)

            if j != 0:
                action = state
                actions.append(action)

        hdf5path = os.path.join(save_path, f"episode_{i}/episode_{i}.hdf5")

        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.array(actions))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(qpos))
            obs.create_dataset("left_arm_dim", data=np.array(left_arm_dim))
            obs.create_dataset("right_arm_dim", data=np.array(right_arm_dim))
            obs.create_dataset("left_arm_state", data=np.array(left_arm_state, dtype=np.float32))
            obs.create_dataset("right_arm_state", data=np.array(right_arm_state, dtype=np.float32))
            obs.create_dataset("perception_arm_dim", data=np.array(perception_arm_dim, dtype=np.float32))
            obs.create_dataset("perception_qpos", data=np.array(perception_qpos, dtype=np.float32))
            image = obs.create_group("images")
            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_wrist_enc, len_right = images_encoding(cam_right_wrist)
            cam_left_wrist_enc, len_left = images_encoding(cam_left_wrist)
            image.create_dataset("cam_high", data=cam_high_enc, dtype=f"S{len_high}")
            image.create_dataset("cam_right_wrist", data=cam_right_wrist_enc, dtype=f"S{len_right}")
            image.create_dataset("cam_left_wrist", data=cam_left_wrist_enc, dtype=f"S{len_left}")

        begin += 1
        print(f"proccess {i} success!")

    return begin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        default="beat_block_hammer",
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("setting", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        default=50,
        help="Number of episodes to process (e.g., 50)",
    )
    args = parser.parse_args()

    task_name = args.task_name
    setting = args.setting
    expert_data_num = args.expert_data_num

    load_dir = os.path.join("../../data", str(task_name), str(setting))

    begin = 0
    print(f'read data from path:{os.path.join("data", load_dir)}')

    target_dir = f"processed_data/{task_name}-{setting}-{expert_data_num}"
    begin = data_transform(
        load_dir,
        expert_data_num,
        target_dir,
    )
