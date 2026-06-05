import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory
import torch

using_yolo_det_model = False
using_yolo_seg_model = True

class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__("yolo_detection_node")

        # 初始化 cv_bridge
        self.bridge = CvBridge()

        self.latest_depth_image_raw = None
        self.latest_depth_image_compressed = None

        # 使用 yolo detection model 位置
        if using_yolo_det_model:
            det_model_path = os.path.join(
                get_package_share_directory("yolo_example_pkg"), "models", "detection.pt"
            )
        
        # 使用 yolo segmentation model 位置
        if using_yolo_seg_model:
            seg_model_path = os.path.join(
                get_package_share_directory("yolo_example_pkg"), "models", "segmentation.pt"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device : ", device)

        # 初始化 YOLO detection 模型
        if using_yolo_det_model:
            self.det_model = YOLO(det_model_path)
            self.det_model.to(device)

        # 初始化 YOLO segmentation 模型
        if using_yolo_seg_model:
            self.seg_model = YOLO(seg_model_path)
            self.seg_model.to(device)

        # 訂閱影像 Topic
        self.image_sub = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.image_callback, 1
        )

        # 訂閱 **無壓縮** 深度圖 Topic
        self.depth_sub_raw = self.create_subscription(
            Image, "/camera/depth/image_raw", self.depth_callback_raw, 1
        )

        # 訂閱 **壓縮** 深度圖 Topic
        self.depth_sub_compressed = self.create_subscription(
            CompressedImage,
            "/camera/depth/compressed",
            self.depth_callback_compressed,
            1,
        )

        # 發佈處理後的影像 Topic
        if using_yolo_det_model:
            self.det_image_pub = self.create_publisher(
                CompressedImage, "/yolo/detection/compressed", 10
            )

        if using_yolo_seg_model:
            self.seg_image_pub = self.create_publisher(
                CompressedImage, "/yolo/segmentation/compressed", 10
            )

        # 發布 目標檢測數據 (是否找到目標 + 距離)
        self.target_pub = self.create_publisher(
            Float32MultiArray, "/yolo/target_info", 10
        )

        self.x_multi_depth_pub = self.create_publisher(
            Float32MultiArray, "/camera/x_multi_depth_values", 10
        )

        # 設定要過濾標籤 (如果為空，那就不過濾)
        self.allowed_labels = {}

        # 設定 YOLO 可信度閾值
        self.conf_threshold = 0.5  # 可以修改這個值來調整可信度

        # 相機畫面中央高度上切成 n 個等距水平點。
        self.x_num_splits = 20

    def depth_callback_raw(self, msg):
        """接收 **無壓縮** 深度圖"""
        try:
            self.latest_depth_image_raw = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        except Exception as e:
            self.get_logger().error(f"Could not convert raw depth image: {e}")

    def depth_callback_compressed(self, msg):
        """接收 **壓縮** 深度圖（當無壓縮深度圖不可用時使用）"""
        try:
            # 自行強制使用 cv2.IMREAD_UNCHANGED 解碼，避開 cv_bridge 的潛在雷區
            np_arr = np.frombuffer(msg.data, np.uint8)
            depth_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                self.latest_depth_image_compressed = depth_img
        except Exception as e:
            self.get_logger().error(f"Could not convert compressed depth image: {e}")

    def image_callback(self, msg):
        """接收影像並進行物體檢測"""
        # 將 ROS 影像消息轉換為 OpenCV 格式
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        if using_yolo_det_model:
            # 使用 YOLO Detection 模型檢測物體
            try:
                det_results = self.det_model(cv_image, conf=self.conf_threshold, verbose=False)
            except Exception as e:
                self.get_logger().error(f"Error during YOLO detection: {e}")
                return
            
            # 繪製 Bounding Box
            det_image = self.draw_bounding_boxes(cv_image, det_results)
            
            # 取得影像中心深度並發布
            self.publish_x_multi_depths(det_image)
            
            # 發佈 Detection 影像
            self.publish_det_image(det_image)

        if using_yolo_seg_model:
            # 使用 YOLO Segmentation 模型檢測物體
            try:
                seg_results = self.seg_model(cv_image, conf=self.conf_threshold, verbose=False)
            except Exception as e:
                self.get_logger().error(f"Error during YOLO segmentation: {e}")
                return

            # 繪製 Mask
            seg_image = self.draw_masks(cv_image, seg_results)
            
            # 發佈 Segmentation 影像
            self.publish_seg_image(seg_image)

    def draw_cross(self, image):
        # 回傳繪製十字架的影像和畫面正中間的像素座標
        height, width = image.shape[:2]
        cx_center = width // 2
        cy_center = height // 2
        # 繪製橫線
        cv2.line(image, (0, cy_center), (width, cy_center), (0, 0, 255), 2)

        # 繪製直線
        cv2.line(
            image,
            (cx_center, cy_center - 10),
            (cx_center, cy_center + 10),
            (0, 0, 255),
            2,
        )

        cv2.line(
            image,
            (cx_center, cy_center - 10),
            (cx_center, cy_center + 10),
            (0, 0, 255),
            2,
        )

        # 計算橫線上的 n 個等分點
        segment_length = width // self.x_num_splits
        points = [
            (i * segment_length, cy_center) for i in range(self.x_num_splits + 1)
        ]  # 11 個點表示 10 段區間的端點

        # 在每個等分點繪製垂直的短黑線
        for x, y in points:
            cv2.line(image, (x, y - 10), (x, y + 10), (0, 0, 0), 2)  # 黑色垂直線

        return image, points

    def draw_bounding_boxes(self, image, results):
        """在影像上繪製 YOLO 檢測到的 Bounding Box"""
        # 一開始預設沒找到目標
        found_target = 0
        target_distance = 0.0
        delta_x = 0.0
        det_image = image.copy()
        tmp_image = image.copy()
        tmp_image, points = self.draw_cross(tmp_image)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_id = int(box.cls[0])
                class_name = self.det_model.names[class_id]

                # # 只保留設定內的標籤
                # if self.allowed_labels and class_name not in self.allowed_labels:
                #     continue

                # 計算 Bounding Box 正中心點
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 如果符合標籤，表示找到目標
                found_target = 1

                # 優先使用無壓縮的深度圖
                depth_value = self.get_depth_at(cx, cy)
                target_distance = depth_value
                depth_text = f"{depth_value:.2f}m" if depth_value else "N/A"

                # ------ 計算與影像中心的偏移量 ------
                delta_x = cx - points[4][0]

                # 根據 class_id 產生隨機但固定的顏色 (B, G, R)
                rng = np.random.RandomState(class_id)
                color = tuple(int(c) for c in rng.randint(0, 256, 3))

                # 繪製框和標籤
                cv2.rectangle(det_image, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f} Depth: {depth_text}"

                cv2.putText(
                    det_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        self.publish_target_info(found_target, target_distance, delta_x)
        return det_image

    def draw_masks(self, image, results):
        """在影像上繪製 YOLO 檢測到的 Mask"""
        height, width = image.shape[:2]
        mask_image = image.copy()  # 從原始影像複製一份來繪製 Mask

        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes
                for i, mask in enumerate(masks):
                    # Create a boolean mask and assign color
                    mask_resized = cv2.resize(mask, (width, height))
                    mask_bool = mask_resized > 0.5
                    
                    # 根據 class_id 產生隨機但固定的顏色 (B, G, R)
                    class_id = int(boxes.cls[i])
                    rng = np.random.RandomState(class_id)
                    color = tuple(int(c) for c in rng.randint(0, 256, 3))
                    
                    # Blend the mask for better visibility
                    mask_colored = np.zeros_like(mask_image)
                    mask_colored[mask_bool] = color
                    mask_image = cv2.addWeighted(mask_image, 1, mask_colored, 0.5, 0)

        return mask_image

    def get_depth_at(self, x, y):
        """
        取得指定像素的深度值，轉換為米 (m)
        若深度出問題，回傳 -1
        """
        # **優先使用無壓縮的深度圖**
        depth_image = (
            self.latest_depth_image_raw
            if self.latest_depth_image_raw is not None
            else self.latest_depth_image_compressed
        )

        if depth_image is None:
            return -1.0

        # 如果深度影像為三通道，那只取第一個數值
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]

        try:
            depth_value = depth_image[y, x]
            if depth_value < 0.0001 or depth_value == 0.0:  # 無效深度
                return -1.0
            return depth_value / 1000.0  # 16-bit 深度圖通常單位為 mm，轉換為 m
        except IndexError:
            return -1.0

    def publish_det_image(self, image):
        """將 Detection 影像轉換並發佈到 ROS"""
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(image)
            self.det_image_pub.publish(compressed_msg)
        except Exception as e:
            self.get_logger().error(f"Could not publish detection image: {e}")

    def publish_seg_image(self, image):
        """將 Segmentation 影像轉換並發佈到 ROS"""
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(image)
            self.seg_image_pub.publish(compressed_msg)
        except Exception as e:
            self.get_logger().error(f"Could not publish segmentation image: {e}")

    def publish_target_info(self, found, distance, delta_x):
        """發佈目標資訊 (找到目標, 距離)"""
        msg = Float32MultiArray()
        msg.data = [float(found), float(distance), float(delta_x)]
        self.target_pub.publish(msg)

    def publish_x_multi_depths(self, image):
        """
        取得畫面 n 個等分點的深度並發布
        """
        height, width = image.shape[:2]
        cy_center = height // 2  # 固定 Y 座標在畫面中心
        segment_length = width // self.x_num_splits

        # 計算 10 個等分點的 X 座標
        points = [(i * segment_length, cy_center) for i in range(self.x_num_splits)]

        # 取得每個等分點的深度值
        depth_values = [self.get_depth_at(x, cy_center) for x, _ in points]

        # 以 Float32MultiArray 發布
        depth_msg = Float32MultiArray()
        depth_msg.data = depth_values
        self.x_multi_depth_pub.publish(depth_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
