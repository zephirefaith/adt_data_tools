import glob
import logging
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPathsProvider,
    AriaDigitalTwinDataProvider,
)
from projectaria_tools.projects.adt import utils as adt_utils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class ADTSequences:
    """Class to load ADT sequences from a root directory. This class is meant to
    be used as the top-level interface for working with ADT. It is meant to
    use ADTSubsequence as a wrapper class for the actual data.
    Variable members:
        - data_root: path to top-level folder containing ADT sequences
        - verbose: whether to print out debug messages
        - device_data: list of lists of paths to ADT sequences, each list contains
                        paths to sequences that are part of the same ADT sequence
        - multi_device_sequences: list of indices of sequences that contain
                                    data from multiple devices
        - single_device_sequences: list of indices of sequences that contain
                                    data from a single device
        - current_sequence: ADTSubSequence object for the current sequence
    """

    def __init__(self, data_root, verbose=False, is_path=False) -> None:
        """Initializes ADTSequences class, data_root can be either a path to a
        top-level folder containing ADT sequences or path to a specific ADT
        Sequence (is_path should be set to True in this case)
        """
        self.data_root = data_root
        self.verbose = verbose
        self.device_data = []
        self._get_sequences(is_path=is_path)
        self._do_indexing()

    def _sanitize_paths(self) -> None:
        """deletes any files from globbed paths"""
        for path in self.file_paths:
            if os.path.isfile(path):
                if self.verbose:
                    logging.debug(f"Removing file {path}")
                self.file_paths.remove(path)

    def _get_sequences(self, is_path=False) -> None:
        """Loads all ADT sequences from data_root, if is_path is True, then
        data_root is a path to a specific ADT sequence"""
        if not is_path:
            self.file_paths = glob.glob(
                os.path.join(self.data_root, "**"), recursive=False
            )
        else:
            self.file_paths = [self.data_root]
        self._sanitize_paths()
        if self.verbose:
            print(f"Found {len(self.file_paths)} folders")
        for i, path in enumerate(self.file_paths):
            path_provider = AriaDigitalTwinDataPathsProvider(path)
            devices = path_provider.get_device_serial_numbers()
            self.device_data.append([])
            for idx, device in enumerate(devices):
                paths = path_provider.get_datapaths_by_device_num(idx)
                self.device_data[i].append(paths)
                if self.verbose:
                    print("Device ID", idx)
                    print("Paths:\n", paths)
        if self.verbose:
            print(f"Loaded data-paths from {len(self.device_data)} sequences")

    def _do_indexing(self) -> None:
        """Creates lists of indices for sequences that contain data from multiple
        devices and sequences that contain data from a single device"""
        self.multi_device_sequences = []
        self.single_device_sequences = []
        for i, data in enumerate(self.device_data):
            if len(data) > 1:
                self.multi_device_sequences.append(i)
            else:
                self.single_device_sequences.append(i)

    def load_sequence(self, index, device_num=0) -> None:
        """Fetches an ADTSubsequence object for the sequence at index and
        device_num"""
        self.data = ADTSubsequence(
            self.device_data[index][device_num], device_num=device_num
        )

    def get_all_annotated_objects(self) -> Tuple[Dict[str, Any], list]:
        return self.data.objects, self.data.object_instance_ids

    def create_object_retrieval_benchmark(self, num_sequences, num_objects):
        """Creates a retrieval benchmark for ADT sequences"""
        num_tot_sequences = len(self.file_paths)
        sampled_sequences = random.sample(range(num_tot_sequences), num_sequences)
        sequence_to_object_map = {}
        for sequence_idx in sampled_sequences:
            self.load_sequence(sequence_idx)
            sampled_objects = random.sample(self.data.object_instance_ids, num_objects)
            sequence_to_object_map[sequence_idx] = sampled_objects
        return sequence_to_object_map


class ADTSubsequence:
    """Wrapper class to make I/O easier for ADT data"""

    def __init__(self, path, device_num=-1, verbose=False) -> None:
        if device_num == -1:
            print("Using dummy subsequence data")
            raise NotImplementedError
        self._verbose = verbose
        self._device_num = device_num
        self.subsequence = AriaDigitalTwinDataProvider(path)
        self._rgb_stream_id = StreamId("214-1")
        self.rgb_timestamps = self.subsequence.get_aria_device_capture_timestamps_ns(
            self._rgb_stream_id
        )
        self._data_keys = [
            "rgb",
            # "slam-l",  # not supported right now
            # "slam-r",  # not supported right now
            "segmentation",
            "2dbbox",
            "3dbbox",
            "pose",
        ]
        self._data_getters = []
        self._data_getters.append(self._get_rgb)
        self._data_getters.append(self._get_segmentation)
        self._data_getters.append(self._get_2dbbox)
        self._data_getters.append(self._get_3dbbox)
        self._data_getters.append(self._get_pose)
        self._load_object_info()
        self._T_device_rgb = (
            self.subsequence.raw_data_provider_ptr()
            .get_device_calibration()
            .get_transform_device_sensor("camera-rgb")
        )

    def _transform_pose(self, data) -> Tuple[Any, int]:
        """Transforms pose from left IMU frame to RGB camera frame"""
        transform_world_rgb = (
            data["pose"][0].transform_scene_device.matrix()
            @ self._T_device_rgb.matrix()
        )
        return (
            transform_world_rgb,
            data["pose"][1],
        )  # this is probably a Sophus SE3 object

    def __process_image_frame(
        self, frame_data, timestamp, frame_name: str
    ) -> Tuple[np.ndarray, int]:
        """Helper function to process image frames and check validity"""
        assert frame_data.is_valid(), "{} not valid at timestamp: {}".format(
            frame_name, timestamp
        )
        if self._verbose:
            print(
                f"Delta b/w given {timestamp*1e-9=} and time of frame {frame_data.dt_ns()*1e-9=}"
            )
        return frame_data.data().to_numpy_array(), frame_data.dt_ns()

    def __process_data_frame(
        self, frame_data, timestamp, frame_name: str
    ) -> Tuple[np.ndarray, int]:
        """Helper function to process data frames and check validity"""
        assert frame_data.is_valid(), "{} not valid at timestamp: {}".format(
            frame_name, timestamp
        )
        if self._verbose:
            print(
                f"Delta b/w given {timestamp*1e-9=} and time of frame {frame_data.dt_ns()*1e-9=}"
            )
        return frame_data.data(), frame_data.dt_ns()

    def __iter__(self):
        return ADTSubsequenceIterator(self)

    def __reversed__(self):
        return ADTSubsequenceIterator(self, reverse=True)

    def _get_rgb(self, timestamp) -> Tuple[np.ndarray, int]:
        """Gets RGB image at timestamp"""
        image = self.subsequence.get_aria_image_by_timestamp_ns(
            timestamp, self._rgb_stream_id
        )
        return self.__process_image_frame(image, timestamp, "RGBImage")

    def _get_segmentation(self, timestamp) -> Tuple[np.ndarray, int]:
        """Gets segmentation annotation image at timestamp from RGB stream"""
        segmentation = self.subsequence.get_segmentation_image_by_timestamp_ns(
            timestamp, self._rgb_stream_id
        )
        return self.__process_image_frame(segmentation, timestamp, "SegmentationImage")

    def _get_2dbbox(self, timestamp) -> Tuple[np.ndarray, int]:
        """Gets GT 2D bounding boxes for all objects at timestamp from RGB stream"""
        bbox_2d = self.subsequence.get_object_2d_boundingboxes_by_timestamp_ns(
            timestamp, self._rgb_stream_id
        )
        return self.__process_data_frame(bbox_2d, timestamp, "2DBoundingBox")

    def _get_3dbbox(self, timestamp) -> Tuple[np.ndarray, int]:
        """Gets GT 3D bounding boxes for all objects at timestamp from RGB stream"""
        bbox_3d = self.subsequence.get_object_3d_boundingboxes_by_timestamp_ns(
            timestamp
        )
        return self.__process_data_frame(bbox_3d, timestamp, "3DBoundingBox")

    def __to_dict(self, an_object) -> Dict[str, Any]:
        """helper function to convert an object to dict"""
        return {k: getattr(an_object, k) for k in an_object.__slots__}

    def _get_intrinsics(self) -> Tuple[Any, int]:
        """Gets camera intrinsics at timestamp"""
        _intrinsics = self.subsequence.get_aria_camera_calibration(self._rgb_stream_id)
        # intrinsics = self.__to_dict(_intrinsics)
        return (_intrinsics, 0)

    def _load_object_info(self):
        """gets all objects in the sequence and creates an ID to Info mapping"""
        self.object_instance_ids = self.subsequence.get_instance_ids()
        self.objects = {}
        for instance_id in self.object_instance_ids:
            self.objects[instance_id] = self.subsequence.get_instance_info_by_id(
                instance_id
            )

    def _get_pose(self, timestamp) -> Tuple[np.ndarray, int]:
        """Gets agent 3D pose at timestamp, this is pose of left IMU"""
        agent_pose = self.subsequence.get_aria_3d_pose_by_timestamp_ns(timestamp)
        return self.__process_data_frame(agent_pose, timestamp, "AgentPose")

    def get_data_by_timestamp(self, timestamp) -> Dict[str, Any]:
        data = {}
        for key, getter in zip(self._data_keys, self._data_getters):
            data[key] = getter(timestamp)

        data["transformed_pose"] = self._transform_pose(data)
        data["intrinsics"] = self._get_intrinsics()
        return data

    def visualize_image_with_3dbbox(self, image, bbox3d, aria3dpose) -> None:
        """Visualizes RGB image with 3D bounding boxes"""
        # get object poses and Aria poses of the selected frame
        print("AABB [xmin, xmax, ymin, ymax, zmin, zmax]: ", bbox3d.aabb)

        # now to project 3D bbox to Aria camera
        # get 6DoF object pose with respect to the target camera
        transform_cam_device = self.subsequence.get_aria_transform_device_camera(
            self._rgb_stream_id
        ).inverse()
        transform_cam_scene = (
            transform_cam_device.matrix()
            @ aria3dpose.transform_scene_device.inverse().matrix()
        )
        transform_cam_obj = transform_cam_scene @ bbox3d.transform_scene_object.matrix()

        # get projection function
        cam_calibration = self.subsequence.get_aria_camera_calibration(
            self._rgb_stream_id
        )
        assert cam_calibration is not None, "no camera calibration"

        # get projected bbox
        reprojected_bbox = adt_utils.project_3d_bbox_to_image(
            bbox3d.aabb, transform_cam_obj, cam_calibration
        )
        if reprojected_bbox:
            # plot
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.axis("off")
            ax.add_patch(
                plt.Polygon(
                    reprojected_bbox,
                    linewidth=1,
                    edgecolor="y",
                    facecolor="none",
                )
            )
            plt.show()
        else:
            print("\033[1m" + "\033[91m" + "Try another object!" + "\033[0m")

    def linear_search_for_object_in_sequence(
        self,
        object_id_list: list,
        num_instances: int = 1,
        stream_id: StreamId = StreamId("214-1"),  # RGB device code
    ):
        """Linear search for object in sequence"""
        found_frames = {}
        iterator = ADTSubsequenceIterator(self, reverse=True)
        num_found = 0
        while True:
            try:
                data = next(iterator)
            except StopIteration:
                print("Reached end of sequence!")
                break
            segmentation_frame = data["segmentation"][0]
            instance_ids_in_frame = set(np.unique(segmentation_frame))
            seg_matches = list(instance_ids_in_frame.intersection(object_id_list))
            instance_ids_with_bbox = set(data["2dbbox"][0].keys())
            bbox_matches = list(instance_ids_with_bbox.intersection(object_id_list))
            if len(seg_matches) != len(bbox_matches):
                print(
                    f"Found {len(seg_matches)} segmentation matches and {len(bbox_matches)} bbox matches! Investigate!"
                )
                breakpoint()

            if seg_matches:
                num_found += 1
                for match in seg_matches:
                    if match not in found_frames:
                        found_frames[match] = data
                    else:
                        found_frames[match].append(data)
            if num_found == num_instances:
                print("Found enough instances! Breaking!")
                break
        return found_frames


class ADTSubsequenceIterator:
    """Iterator class for ADTSubSequence (only operates over RGB data right now)"""

    def __init__(self, container: ADTSubsequence, reverse=False) -> None:
        self.container = container
        self._ns_delta = int(0.1 * 1e9)
        self._threshold_ns = int(0.1 * 1e9)
        if reverse:
            self.end_limit = self.container.rgb_timestamps[0]
            self.start_limit = self.container.rgb_timestamps[-1]
            self._ns_delta *= -1
        else:
            self.end_limit = self.container.rgb_timestamps[-1]
            self.start_limit = self.container.rgb_timestamps[0]
        self.curr_ts = self.start_limit
        self._reverse = reverse

    def __next__(self):
        while True:
            if self._reverse and self.curr_ts <= self.end_limit:
                raise StopIteration
            elif not self._reverse and self.curr_ts >= self.end_limit:
                raise StopIteration
            data = self.container.get_data_by_timestamp(self.curr_ts)
            self.curr_ts += self._ns_delta
            if abs(data["segmentation"][1]) < self._threshold_ns:
                break
        return data


@click.command()
@click.option("--adt-path", type=click.Path(exists=True))
@click.option("--is-root/--not-root", type=bool, default=True, is_flag=True)
@click.option("--verbose/--not-verbose", type=bool, default=False, is_flag=True)
@click.option("--visualize/--no-visualize", type=bool, default=True, is_flag=True)
@click.option("--fwd/--reverse", type=bool, default=True, is_flag=True)
def main(adt_path: str, is_root: bool, verbose: bool, visualize: bool, fwd: bool):
    adt_sequences = ADTSequences(adt_path, is_path=(not is_root), verbose=verbose)
    sequence_index = 10
    subsequence_index = 0
    adt_sequences.load_sequence(sequence_index, subsequence_index)
    if fwd:
        print("Trying fwd access")
    else:
        print("Trying reverse access")
    iterator = ADTSubsequenceIterator(adt_sequences.data, reverse=(not fwd))
    data = next(iterator)
    print(data)
    all_object_ids = np.unique(data["segmentation"][0])
    all_object_ids = all_object_ids[all_object_ids != 0]
    random_object_id = np.random.choice(all_object_ids)
    object_info, _ = adt_sequences.get_all_annotated_objects()
    print("Object info: ", object_info[random_object_id])
    print(f"Aria pose: {data['pose'][0].transform_scene_device.matrix()}")
    if visualize:
        plt.imshow(data["rgb"][0])
        plt.show()
        plt.imshow(data["segmentation"][0])
        plt.show()
        adt_sequences.data.visualize_image_with_3dbbox(
            data["rgb"][0],
            data["3dbbox"][0][random_object_id],
            data["pose"][0],
        )
        adt_sequences.data.visualize_image_with_3dbbox(
            data["segmentation"][0],
            data["3dbbox"][0][random_object_id],
            data["pose"][0],
        )


if __name__ == "__main__":
    main()
