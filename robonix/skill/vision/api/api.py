import numpy as np
from ultralytics import YOLOE
import traceback
import os
import cv2
import random
from datetime import datetime
from .vision import px2xy, remove_mask_outliers, get_mask_center_opencv

from robonix.manager.eaios_decorators import eaios

def create_color_palette(n_colors):
    colors = []
    for i in range(n_colors):
        hue = (i * 137.508) % 360
        saturation = 70 + random.randint(-20, 20)
        value = 80 + random.randint(-20, 20)

        hsv = np.array([[[hue, saturation, value]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        colors.append(bgr[0, 0].tolist())
    return colors

def draw_rounded_rectangle(img, x1, y1, x2, y2, color, thickness=2, radius=10):
    # Draw main rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Draw rounded corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)

def put_text_with_background(
    img,
    text,
    position,
    font_scale=0.6,
    thickness=2,
    bg_color=(0, 0, 0),
    text_color=(255, 255, 255),
):
    font = cv2.FONT_HERSHEY_DUPLEX
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    x, y = position
    padding = 8

    # Draw background rectangle
    bg_rect = [
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
    ]
    cv2.rectangle(img, bg_rect[0], bg_rect[1], bg_color, -1)

    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

@eaios.api
@eaios.caller
def skl_detect_objs(self_entity, camera_name: str) -> dict:
    """
    Detect all objects in the current view of the specified camera and return their categories and coordinates (in the 'map' frame).

    Changes vs. previous version:
      1) BBox drawing is guaranteed ("bbox-first"): draw/clamp bbox + basic label BEFORE any mask/depth/TF logic.
      2) Clamp bbox to image bounds; skip degenerate boxes with explicit warning.
      3) Use cv2.rectangle as a hard fallback (always visible), while keeping draw_rounded_rectangle as optional.
      4) Post-processing (mask center, depth, px2xy, TF) is isolated; failures will NOT suppress bbox drawing.

    Args:
        camera_name: Name of the camera (e.g., 'camera')
    Returns:
        Dict: {obj_name: (x, y, z)} - Mapping from object name to 3D coordinates in the 'map' frame
    """
    try:
        # Get RGB and depth images
        rgb_image, depth_image = self_entity.cap_camera_dep_rgb(camera_name=camera_name)
        if rgb_image is None or depth_image is None:
            print("Failed to get RGB and depth images")
            return {}

        # Get camera parameters
        camera_info = self_entity.cap_camera_info(camera_name=camera_name)
        if camera_info is None:
            print("Failed to get camera info")
            return {}

        np_color_image = np.array(rgb_image, dtype=np.uint8)
        np_depth_image = np.array(depth_image, dtype=np.uint16)
        if np_color_image is None:
            return {}

        # Load YOLO model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "..", "models", "yoloe-11l-seg-pf.pt")
        yolo = YOLOE(model_path)

        # Run YOLO detection
        results = yolo(source=np_color_image, device="cuda:0")
        detection = results[0] if results else None
        if detection is None:
            return {}

        # Extract detection results
        object_boxes = detection.boxes.xyxy.cpu().numpy()
        n_objects = int(object_boxes.shape[0])
        masks = detection.masks.cpu() if getattr(detection, "masks", None) is not None else None
        detection_class = detection.boxes.cls.cpu().numpy()
        detection_conf = detection.boxes.conf.cpu().numpy()
        detected_objects = {}

        # Create visualization
        colors = create_color_palette(max(n_objects, 1))
        vis_image = np_color_image.copy()

        overlay = vis_image.copy()
        cv2.rectangle(overlay, (0, 0), (vis_image.shape[1], 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0, vis_image)

        detected_count = 0
        print(f"YOLO detected {n_objects} objects total")

        img_h, img_w = vis_image.shape[:2]

        for i in range(n_objects):
            # ---- basic per-object info (should not throw) ----
            try:
                cls_id = int(detection_class[i])
                name = detection.names[cls_id] if hasattr(detection, "names") else str(cls_id)
                conf = float(detection_conf[i])
            except Exception:
                name = f"obj_{i}"
                conf = float(detection_conf[i]) if i < len(detection_conf) else 0.0

            print(f"Object {i}: {name} with confidence {conf:.3f}")

            if conf < 0.2:
                print(f"Skipping {name} due to low confidence {conf:.3f}")
                continue

            detected_count += 1

            # =========================================================
            # 1) BBOX-FIRST: ALWAYS draw bbox + a minimal label first
            # =========================================================
            x1, y1, x2, y2 = object_boxes[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Clamp to image bounds (critical)
            x1 = max(0, min(x1, img_w - 1))
            x2 = max(0, min(x2, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            y2 = max(0, min(y2, img_h - 1))

            if x2 <= x1 or y2 <= y1:
                print(f"[WARN] Invalid bbox for {name}: {(x1, y1, x2, y2)}")
                continue

            color = colors[i % len(colors)]

            # Hard fallback bbox (always visible)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness=3)

            # Optional rounded bbox (allowed to fail without suppressing bbox)
            try:
                draw_rounded_rectangle(vis_image, x1, y1, x2, y2, color, thickness=3)
            except Exception as e:
                print(f"[WARN] draw_rounded_rectangle failed for {name}: {e}")

            # Minimal label (always present)
            try:
                put_text_with_background(
                    vis_image,
                    f"{name} ({conf:.2f})",
                    (x1, max(y1 - 10, 30)),
                    font_scale=0.55,
                    thickness=1,
                    bg_color=color,
                    text_color=(255, 255, 255),
                )
            except Exception as e:
                print(f"[WARN] put_text_with_background failed for {name}: {e}")

            # =========================================================
            # 2) Post-processing: mask/depth/px2xy/TF (may fail safely)
            # =========================================================
            try:
                # If masks are missing, we can still attempt a bbox-center based estimate
                if masks is None or getattr(masks, "xy", None) is None or i >= len(masks.xy):
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    single_selection_mask = None
                else:
                    mask_points = masks.xy[i].reshape(-1, 1, 2).astype(np.int32)
                    center_x, center_y = get_mask_center_opencv(mask_points)
                    single_selection_mask = np.array(masks.xy[i])

                # Guard center within image bounds
                center_x = max(0, min(int(center_x), img_w - 1))
                center_y = max(0, min(int(center_y), img_h - 1))

                # Compute object depth
                depths = []
                if single_selection_mask is not None:
                    for point in single_selection_mask:
                        p_x = int(point[0])
                        p_y = int(point[1])
                        if 0 <= p_x < np_depth_image.shape[1] and 0 <= p_y < np_depth_image.shape[0]:
                            depth = np_depth_image.item(p_y, p_x)
                            if not (depth == 0 or depth != depth):
                                depths.append(depth)

                selected_depth = remove_mask_outliers(depths, lower_percentile=10, upper_percentile=70) if len(depths) else []
                avg_depth = (sum(selected_depth) / len(selected_depth)) if len(selected_depth) else 0.0
                avg_depth = avg_depth / 1000.0  # meters

                center_depth = float(np_depth_image[center_y, center_x]) / 1000.0
                if center_depth == 0 or center_depth != center_depth:
                    center_depth = avg_depth

                # If depth is still unusable, keep bbox but skip 3D coordinate generation
                if center_depth == 0 or center_depth != center_depth:
                    print(f"[WARN] Depth invalid for {name}. bbox drawn, but skipping 3D coords.")
                    continue

                # Convert pixel to camera coordinates
                world_x, world_y = px2xy([center_x, center_y], camera_info["k"], camera_info["d"], center_depth)

                # TF transform (your original code uses camera_link -> camera_link; kept as-is)
                map_x, map_y, map_z = self_entity.cap_tf_transform(
                    source_frame='camera_link',
                    target_frame='camera_link',
                    x=world_x,
                    y=world_y,
                    z=center_depth
                )
                detected_objects[name] = (map_x, map_y, map_z, conf)

                # Upgrade label with full info (best-effort; won't affect bbox)
                label_full = f"{name} ({conf:.4f}) | D:{center_depth:.4f}m | ({map_x:.4f}, {map_y:.4f})"
                try:
                    put_text_with_background(
                        vis_image,
                        label_full,
                        (x1, max(y1 - 10, 30)),
                        font_scale=0.5,
                        thickness=1,
                        bg_color=color,
                        text_color=(255, 255, 255),
                    )
                except Exception as e:
                    print(f"[WARN] full label draw failed for {name}: {e}")

                print(
                    f"object {name}: depth={center_depth:.3f}m, "
                    f"pixel_center=({center_x}, {center_y}), "
                    f"map_pos=({map_x:.3f}, {map_y:.3f}, {map_z:.3f})"
                )

            except Exception as e:
                print(f"[WARN] Postprocess failed for object {i} ({name}): {str(e)}")
                print(f"Traceback: {traceback.format_exc()}")
                detected_objects[name] = ((x1 + x2) / 2, (y1 + y2) / 2, 0.0, conf)
                # bbox already drawn; continue to next object
                continue

        # If nothing passed threshold, make that explicit on the saved image
        if detected_count == 0:
            try:
                put_text_with_background(
                    vis_image,
                    "No detections above threshold (conf>=0.7)",
                    (10, 30),
                    font_scale=0.8,
                    thickness=2,
                    bg_color=(0, 0, 255),
                    text_color=(255, 255, 255),
                )
            except Exception:
                pass

        # Add timestamp at bottom right corner
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        img_height, img_width = vis_image.shape[:2]

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(timestamp, font, font_scale, thickness)

        padding = 10
        text_x = img_width - text_width - padding
        text_y = img_height - padding

        put_text_with_background(
            vis_image,
            timestamp,
            (text_x, text_y),
            font_scale=font_scale,
            thickness=thickness,
            bg_color=(30, 30, 30),
            text_color=(200, 200, 200),
        )

        # Save visualization
        output_dir = os.path.join(script_dir, "..", "output")
        os.makedirs(output_dir, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"detection_result_{timestamp_str}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        cv2.imwrite(output_path, vis_image)
        print(f"detection visualization saved to: {output_path}")
        print("detected_objects:", detected_objects)

        return detected_objects

    except Exception as e:
        print(f"Error in skl_detect_objs: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return {}
