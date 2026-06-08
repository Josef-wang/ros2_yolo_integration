import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray, String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory
import torch

using_yolo_det_model = True
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

        # auto-task 目標 class：只有此 class 的 box 會被當成候選目標。
        # None 代表不過濾 (回報任意 class)。由 /yolo/target_class 動態切換，切 task 不用重啟。
        self.target_class_name = None
        self.target_class_sub = self.create_subscription(
            String, "/yolo/target_class", self.target_class_callback, 10
        )

        # delta_x 的對齊基準 x (以畫面寬度比例表示)。
        # 0.5 = 畫面正中央，使 delta_x≈0 時目標正對車頭/夾爪 → APPROACH 會直接前進。
        # (舊版寫死 points[4]≈0.2 會讓正前方的熊算出大 delta_x，車子只會原地轉不前進。)
        # 若夾爪在畫面裡偏左/右，再微調此值對準夾爪實際橫向位置。
        self.align_ref_x_ratio = 0.5

        # 設定 YOLO 可信度閾值
        self.conf_threshold = 0.35  # 近距離抓取穩定 vs 移動掉框 的折衷；移動掉框靠 lost_grace+dedup 兜底

        # ---- 目標鎖定 (hysteresis) ----
        # 邊緣的熊 bbox 會閃爍、多隻熊時選擇也會逐幀跳動。鎖住「離上一幀選中 cx 最近」
        # 的候選，避免目標身分亂跳。lock_gate_ratio = 容許的橫向位移 (畫面寬比例)。
        self.last_target_cx = None
        self.lock_gate_ratio = 0.20     # 上一幀 cx ±20% 寬內視為同一隻
        # 黏鎖：鎖定的熊在接近時會離開視野，此時別跳去追畫面上另一隻(遠處)。
        # gate 內連續找不到原目標的容忍幀數，超過才解除鎖定、全域重挑。
        # 容忍期內回報 found=0，讓 pros_car 的 lost_grace 往原方向回找。
        self.lock_miss_limit = 5
        self._lock_miss = 0
        # 面積下限 (畫面面積比例)：濾掉太小的閃爍弱框。
        self.min_area_ratio = 0.0005    # ~0.05% 影像面積
        # 去重：同一隻熊的重疊重複框，IoU 超過此值視為同一隻只留一個。
        self.dedup_iou = 0.5
        # 距離閘門：已知深度且 > 此值(公尺)的遠熊直接剔除候選 → 不會被選成 TARGET。
        # 只擋「有效深度且確實遠」者；depth<=0(邊緣近熊常見的未知深度)保留。
        self.max_target_distance = 1.5

        # 相機畫面中央高度上切成 n 個等距水平點。
        self.x_num_splits = 20

    def _dedup_overlapping(self, candidates):
        """同一隻熊常被吐出多個重疊框 → IoU > dedup_iou 視為同一隻，貪婪保留面積最大者。"""
        kept = []
        for c in sorted(candidates, key=lambda c: c["area"], reverse=True):
            x1, y1, x2, y2 = c["box"]
            dup = False
            for k in kept:
                kx1, ky1, kx2, ky2 = k["box"]
                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                union = c["area"] + k["area"] - inter
                if union > 0 and inter / union > self.dedup_iou:
                    dup = True
                    break
            if not dup:
                kept.append(c)
        return kept

    def target_class_callback(self, msg):
        """動態設定要追蹤的目標 class (task1=bear, task3=knob...)；空字串代表不過濾。"""
        name = msg.data.strip()
        self.target_class_name = name if name else None
        # 重置目標鎖定：避免上一輪 task 殘留的 last_target_cx 讓新一輪「鎖」在舊位置、
        # 用 cx 連續性挑到遠熊而非面積最大的近熊 (每次按 s 啟動 task 都會重發 class)。
        self.last_target_cx = None
        self._lock_miss = 0
        self.get_logger().info(f"YOLO target class set to: {self.target_class_name} (lock reset)")

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
        """在影像上繪製 YOLO 檢測到的 Bounding Box，並挑選「單一」目標發布。

        目標選擇：
          1. 先依 self.target_class_name 過濾候選 (None 代表不過濾)。
          2. 濾掉面積 < min_area_ratio 的閃爍弱框。
          3. 目標鎖定：若上一幀有選中，優先在其 cx 鄰近的候選裡挑 (咬住同一隻)。
          4. 主排序「bbox 面積最大 = 最近」(熊同尺寸，且面積不像中心 depth 會在邊緣失效)。
        被選中的目標才會寫進 /yolo/target_info；depth 只用於回報抓取距離。
        """
        found_target = 0
        target_distance = 0.0
        delta_x = 0.0
        det_image = image.copy()

        height, width = det_image.shape[:2]
        # delta_x 對齊基準 x：可由 align_ref_x_ratio 調整 (預設等效舊版 points[4])
        ref_x = int(self.align_ref_x_ratio * width)

        # ---- 1. 收集符合目標 class 的候選 box ----
        candidates = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.det_model.names[class_id]

                # 只保留目標 class (None 代表不過濾)
                if self.target_class_name and class_name != self.target_class_name:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                candidates.append(
                    {
                        "box": (x1, y1, x2, y2),
                        "cx": cx,
                        "cy": cy,
                        "depth": self.get_depth_at(cx, cy),  # 無效時為 -1.0
                        "area": (x2 - x1) * (y2 - y1),
                        "conf": float(box.conf),
                        "class_id": class_id,
                        "class_name": class_name,
                    }
                )

        # ---- 2. 挑選單一目標 ----
        # 未鎖定時：挑 bbox 面積最大 (=最近，熊同尺寸)。
        # 鎖定後：一律追「離上次 cx 最近」的同一隻，**永不用全域面積重挑**，
        #   這樣近熊靠到邊緣被裁切(面積變小)時，仍是離上次最近的 → 絕不跳去畫面另一端的遠熊。
        chosen = None
        min_area = self.min_area_ratio * (width * height)
        candidates = [c for c in candidates if c["area"] >= min_area]
        # 去重：同一隻熊常被吐出多個重疊框 (尤其 conf 低時)，會害鎖定在框間跳動。
        # 重疊 (IoU > dedup_iou) 視為同一隻，只留面積最大者。
        candidates = self._dedup_overlapping(candidates)
        # 距離閘門：剔除「有效深度且 > max_target_distance」的遠熊；保留 depth<=0(未知)的近邊緣熊。
        candidates = [c for c in candidates
                      if not (c["depth"] > 0.0 and c["depth"] > self.max_target_distance)]
        if not candidates:
            self.last_target_cx = None  # 本幀無候選 → 解除鎖定
            self._lock_miss = 0
        elif self.last_target_cx is None:
            # 還沒鎖定 → 挑面積最大 (最近) 當目標並鎖定
            chosen = max(candidates, key=lambda c: c["area"])
            self.last_target_cx = chosen["cx"]
            self._lock_miss = 0
        else:
            # 已鎖定 → 追離上次 cx 最近的候選 (連續性追蹤)
            gate = self.lock_gate_ratio * width
            nearest = min(candidates, key=lambda c: abs(c["cx"] - self.last_target_cx))
            if abs(nearest["cx"] - self.last_target_cx) <= gate or self._lock_miss >= self.lock_miss_limit:
                # gate 內 = 同一隻；或已跟丟夠久 → 接受最靠近上次位置者 (仍非遠熊)
                chosen = nearest
                self.last_target_cx = chosen["cx"]
                self._lock_miss = 0
            else:
                # 最近的候選也離上次太遠 → 暫時跟丟，回報 found=0 讓 lost_grace 回找
                self._lock_miss += 1
                chosen = None

        # ---- 3. 繪製所有候選 (被選中的用粗框 + TARGET 標示) ----
        for c in candidates:
            x1, y1, x2, y2 = c["box"]
            rng = np.random.RandomState(c["class_id"])
            color = tuple(int(v) for v in rng.randint(0, 256, 3))
            is_chosen = c is chosen
            cv2.rectangle(det_image, (x1, y1), (x2, y2), color, 3 if is_chosen else 1)
            depth_text = f"{c['depth']:.2f}m" if c["depth"] > 0.0 else "N/A"
            tag = "TARGET " if is_chosen else ""
            label = f"{tag}{c['class_name']} {c['conf']:.2f} Depth: {depth_text}"
            cv2.putText(
                det_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # ---- 4. 設定要回報的目標資訊 ----
        if chosen is not None:
            found_target = 1
            # depth 無效 (面積 fallback 情況) 統一回 -1.0，與「過近」語意一致
            target_distance = chosen["depth"] if chosen["depth"] > 0.0 else -1.0
            delta_x = float(chosen["cx"] - ref_x)

        # 畫出對齊基準線，方便在 Foxglove 校準
        cv2.line(det_image, (ref_x, 0), (ref_x, height), (0, 255, 255), 1)

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
