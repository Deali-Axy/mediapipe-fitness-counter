from concurrent import futures
import logging
import base64
import grpc
import cv2
import numpy as np

import pose_embedding as pe  # 姿态关键点编码模块
import pose_classifier as pc  # 姿态分类器
import results_mooth as rs  # 分类结果平滑
import counter  # 动作计数器
import visualizer as vs  # 可视化模块

from io import BytesIO
from PIL import Image
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

from grpc_apis import video_frame_pb2
from grpc_apis import video_frame_pb2_grpc


def pil_image_to_base64(img: Image.Image):
    """Pillow Image to base64"""
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf8')
    return base64_str


def bytes_to_base64(image_bytes):
    """bytes 转 base64"""
    image_base4 = base64.b64encode(image_bytes).decode('utf8')
    return image_base4


def numpy_to_base64(image_np):
    """numpy 转 base64"""
    data = cv2.imencode('.jpg', image_np)[1]
    image_bytes = data.tobytes()
    image_base4 = base64.b64encode(image_bytes).decode('utf8')
    return image_base4


def base64_to_bytes(image_base64):
    """base64 转 bytes"""
    image_bytes = base64.b64decode(image_base64)
    return image_bytes


def base64_to_numpy(image_base64):
    """base64转数组"""
    image_bytes = base64.b64decode(image_base64)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np2


class VideoCapture(video_frame_pb2_grpc.VideoCaptureServicer):
    def __init__(self, mode):
        """
        VideoCapture init

        :param mode: 1.俯卧撑; 2.深蹲; 3.引体向上
        """
        super().__init__()
        self.mode = mode
        if mode == 1:
            self.class_name = 'push_down'
        elif mode == 2:
            self.class_name = 'squat_down'
        elif mode == 3:
            self.class_name = 'pull_up'

        # Initilize tracker, classifier and counter.
        # Do that before every video as all of them have state.

        # Folder with pose class CSVs. That should be the same folder you using while
        # building classifier to output CSVs.
        self.pose_samples_folder = 'fitness_poses_csvs_out'

        # Initialize tracker.
        self.pose_tracker = mp_pose.Pose()

        # Initialize embedder.
        self.pose_embedder = pe.FullBodyPoseEmbedder()

        # Initialize classifier.
        # Check that you are using the same parameters as during bootstrapping.
        self.pose_classifier = pc.PoseClassifier(
            pose_samples_folder=self.pose_samples_folder,
            pose_embedder=self.pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10
        )

        # # Uncomment to validate target poses used by classifier and find outliers.
        # outliers = pose_classifier.find_pose_sample_outliers()
        # print('Number of pose sample outliers (consider removing them): ', len(outliers))

        # Initialize EMA smoothing.
        self.pose_classification_filter = rs.EMADictSmoothing(
            window_size=10,
            alpha=0.2
        )

        # Initialize counter.
        self.repetition_counter = counter.RepetitionCounter(
            class_name=self.class_name,
            enter_threshold=5,
            exit_threshold=4
        )

        # Initialize renderer.
        self.pose_classification_visualizer = vs.PoseClassificationVisualizer(
            class_name=self.class_name,
            # plot_x_max=video_n_frames,
            # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
            plot_y_max=10
        )

    def __del__(self):
        self.pose_tracker.close()

    def GetResultBuffer(self, request: video_frame_pb2.InputFrame, context):
        # Run pose tracker.
        input_frame = base64_to_numpy(request.base64buffer)
        result = self.pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks

        # Draw pose prediction.
        output_frame = input_frame.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS
            )

        if pose_landmarks is not None:
            # Get landmarks.
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                                       for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

            # Classify the pose on the current frame.
            pose_classification = self.pose_classifier(pose_landmarks)

            # Smooth classification using EMA.
            pose_classification_filtered = self.pose_classification_filter(pose_classification)

            # Count repetitions.
            repetitions_count = self.repetition_counter(pose_classification_filtered)
        else:
            # No pose => no classification on current frame.
            pose_classification = None

            # Still add empty classification to the filter to maintaing correct
            # smoothing for future frames.
            pose_classification_filtered = self.pose_classification_filter(dict())
            pose_classification_filtered = None

            # Don't update the counter presuming that person is 'frozen'. Just
            # take the latest repetitions count.
            repetitions_count = self.repetition_counter.n_repeats

        output_frame = cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR)

        # Draw classification plot and repetition counter.
        output_img: Image.Image = self.pose_classification_visualizer(
            frame=output_frame,
            pose_classification=pose_classification,
            pose_classification_filtered=pose_classification_filtered,
            repetitions_count=repetitions_count
        )

        return video_frame_pb2.OutputFrame(base64buffer=pil_image_to_base64(output_img))


def serve():
    port = '50055'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    video_frame_pb2_grpc.add_VideoCaptureServicer_to_server(VideoCapture(2), server)
    server.add_insecure_port('[::]:' + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
