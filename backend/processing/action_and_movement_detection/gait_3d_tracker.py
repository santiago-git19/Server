"""
Gait3DTracker
=============

Clase para reconstruir la trayectoria 3D del centro de cadera y
acumular la distancia recorrida a partir de:
- Keypoints 2D procesados (recibidos desde otra clase)
- Frames de profundidad correspondientes
- Parámetros intrínsecos de la cámara para proyección 3D

Uso rápido:

from backend.processing.action_and_movement_detection.gait_3d_tracker import Gait3DTracker

# Parámetros intrínsecos de la cámara (ejemplo para Orbbec Gemini 335Le)
intrinsics = {
    'fx': 570.3, 'fy': 570.3, 'cx': 320.0, 'cy': 240.0
}

tracker = Gait3DTracker(camera_intrinsics=intrinsics)

# Procesar chunk con keypoints y frames de profundidad:
distance = tracker.process_chunk(keypoints_list, depth_frames)
print(f"Distancia total: {tracker.total_distance_m} metros")

La trayectoria (en metros) queda en tracker.trajectory_m (lista de np.ndarray de shape (3,)).
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Any, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Gait3DTracker:
    """
    Reconstruye la trayectoria 3D del punto medio de la cadera y acumula distancia.

    Parámetros:
    - camera_intrinsics: diccionario con parámetros intrínsecos {'fx', 'fy', 'cx', 'cy'}
    - region: tamaño (en píxeles) del lado de la ventana cuadrada alrededor del hip para promediar depth
    - min_conf: confianza mínima de cada cadera (COCO 11 y 12) para considerar el punto
    - smoothing_alpha: factor de suavizado EMA (0..1), mayor = más suave
    - min_valid_depth_percent: porcentaje mínimo de píxeles válidos en la región para usar su mediana
    - max_jump_m: salto máximo permitido entre frames (en metros) para descartar outliers
    """

    COCO_LEFT_HIP = 11
    COCO_RIGHT_HIP = 12

    def __init__(
        self,
        camera_intrinsics: Dict[str, float],
        region: int = 7,
        min_conf: float = 0.3,
        smoothing_alpha: float = 0.2,
        min_valid_depth_percent: float = 25.0,
        max_jump_m: float = 1.0,
    ) -> None:
        if not camera_intrinsics or not all(k in camera_intrinsics for k in ['fx', 'fy', 'cx', 'cy']):
            raise ValueError("camera_intrinsics debe contener las claves: fx, fy, cx, cy")
        
        self.camera_intrinsics = camera_intrinsics
        self.region = max(3, int(region) | 1)  # forzar impar >=3
        self.min_conf = float(min_conf)
        self.alpha = float(smoothing_alpha)
        self.min_valid_percent = float(min_valid_depth_percent)
        self.max_jump_m = float(max_jump_m)

        self.reset()

    def reset(self) -> None:
        self.trajectory_m: List[np.ndarray] = []
        self.total_distance_m: float = 0.0
        self._last_filtered: Optional[np.ndarray] = None

    @staticmethod
    def _extract_hip_center_2d(
        keypoints: Optional[List[Tuple[float, float, float, int]]],
        min_conf: float,
    ) -> Optional[Tuple[int, int, float]]:
        """
        Devuelve (x, y, conf) del centro de cadera usando COCO 11 (left_hip) y 12 (right_hip).
        keypoints: lista de tuplas (x, y, conf, part_id)
        """
        if not keypoints:
            return None

        left_hip = None
        right_hip = None
        for x, y, conf, pid in keypoints:
            if conf < min_conf:
                continue
            if pid == Gait3DTracker.COCO_LEFT_HIP:
                left_hip = (float(x), float(y), float(conf))
            elif pid == Gait3DTracker.COCO_RIGHT_HIP:
                right_hip = (float(x), float(y), float(conf))

        if left_hip and right_hip:
            x_c = int(round((left_hip[0] + right_hip[0]) / 2.0))
            y_c = int(round((left_hip[1] + right_hip[1]) / 2.0))
            conf_c = float(min(left_hip[2], right_hip[2]))
            return (x_c, y_c, conf_c)

        return None

    def _region_depth_mm(self, depth_frame: np.ndarray, x: int, y: int) -> Optional[float]:
        """Obtiene la profundidad mediana en una región alrededor del punto especificado"""
        try:
            h, w = depth_frame.shape
            half = self.region // 2
            
            # Asegurar que la región esté dentro de la imagen
            x1 = max(0, x - half)
            y1 = max(0, y - half)
            x2 = min(w, x + half + 1)
            y2 = min(h, y + half + 1)
            
            region = depth_frame[y1:y2, x1:x2]
            valid_depths = region[region > 0]
            
            if len(valid_depths) >= len(region.flatten()) * (self.min_valid_percent / 100.0):
                return float(np.median(valid_depths))
            
            # Fallback a píxel central
            if 0 <= x < w and 0 <= y < h and depth_frame[y, x] > 0:
                return float(depth_frame[y, x])
                
            return None
        except Exception as e:
            logger.error(f"Error obteniendo profundidad en región: {e}")
            return None
    
    def _convert_to_3d_coordinates(self, x: int, y: int, z_mm: float) -> Optional[Tuple[float, float, float]]:
        """Convierte coordenadas 2D + profundidad a coordenadas 3D usando parámetros intrínsecos"""
        try:
            fx = self.camera_intrinsics['fx']
            fy = self.camera_intrinsics['fy']
            cx = self.camera_intrinsics['cx']
            cy = self.camera_intrinsics['cy']
            
            # Convertir de píxeles a coordenadas 3D en mm
            x_3d = (x - cx) * z_mm / fx
            y_3d = (y - cy) * z_mm / fy
            z_3d = z_mm
            
            return (x_3d, y_3d, z_3d)
            
        except Exception as e:
            logger.error(f"Error convirtiendo a coordenadas 3D: {e}")
            return None

    def update(
        self,
        keypoints: Optional[List[Tuple[float, float, float, int]]],
        depth_frame: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Actualiza el tracker con un nuevo frame y acumula distancia.

        Args:
            keypoints: lista (x, y, conf, part_id) de TRT Pose
            depth_frame: frame de profundidad (mm) correspondiente al color

        Returns:
            np.ndarray(3,) con el punto filtrado en metros (X,Y,Z) o None si inválido
        """
        if depth_frame is None or keypoints is None:
            print("++++++++++++++++++"+ "depth_frame is None or keypoints is None:" +"++++++++++++++++++")
            return None

        # 1) Centro de cadera 2D
        hip = self._extract_hip_center_2d(keypoints, self.min_conf)
        if not hip:
            print("++++++++++++++++++"+ "not hip" +"++++++++++++++++++")
            return None
        x, y, _ = hip

        # 2) Profundidad robusta (mm)
        z_mm_region = self._region_depth_mm(depth_frame, x, y)
        if z_mm_region is None:
            print("++++++++++++++++++"+ "not z_mm_region" +"++++++++++++++++++")
            return None

        # 3) Coordenadas 3D en mm
        xyz_mm = self._convert_to_3d_coordinates(x, y, z_mm_region)
        if xyz_mm is None:
            print("++++++++++++++++++"+ "not xyz_mm" +"++++++++++++++++++")
            return None

        x_3d, y_3d, z_3d = xyz_mm
        
        # 4) A metros
        point_m = np.array([x_3d, y_3d, z_3d], dtype=np.float32) / 1000.0

        # 5) Suavizado EMA
        if self._last_filtered is None:
            filtered = point_m
        else:
            filtered = self.alpha * point_m + (1.0 - self.alpha) * self._last_filtered

        # 6) Acumular distancia (y rechazo de outliers)
        if len(self.trajectory_m) > 0:
            step = float(np.linalg.norm(filtered - self.trajectory_m[-1]))
            if step <= self.max_jump_m:
                self.total_distance_m += step

        self.trajectory_m.append(filtered)
        self._last_filtered = filtered
        return filtered

    def process_chunk(
        self, 
        keypoints_list: List[Optional[List[Tuple[float, float, float, int]]]], 
        depth_frames: List[np.ndarray]
    ) -> float:
        """
        Procesa un chunk de keypoints y frames de profundidad
        
        Args:
            keypoints_list: Lista de keypoints por frame (ya procesados por otra clase)
            depth_frames: Lista de frames de profundidad correspondientes
            
        Returns:
            float: Distancia acumulada en este chunk (metros)
        """
        if len(keypoints_list) != len(depth_frames):
            logger.error("Las listas de keypoints y frames de profundidad deben tener el mismo tamaño")
            return 0.0
        
        initial_distance = self.total_distance_m
        processed_frames = 0
        
        for i, (keypoints, depth_frame) in enumerate(zip(keypoints_list, depth_frames)):
            try:
                point_3d = self.update(keypoints, depth_frame)
                if point_3d is not None:
                    processed_frames += 1
                    logger.debug(f"Frame {i}: Punto 3D procesado: {point_3d}")
                else:
                    logger.debug(f"Frame {i}: No se pudo obtener punto 3D válido")
                    print("++++++++++++++++++"+ f"Frame {i}: No se pudo obtener punto 3D válido" +"++++++++++++++++++") 
            except Exception as e:
                logger.error(f"Error procesando frame {i}: {e}")
                continue
        
        chunk_distance = self.total_distance_m - initial_distance
        logger.info(f"Chunk procesado: {processed_frames}/{len(keypoints_list)} frames válidos, "
                   f"distancia acumulada: {chunk_distance:.3f} metros")
        
        return chunk_distance
    
    def process_chunk_from_server_data(
        self,
        chunk_data: Dict[str, Any]
    ) -> float:
        """
        Procesa un chunk de datos recibido del servidor
        
        Args:
            chunk_data: Diccionario con los datos del chunk que debe contener:
                       - 'keypoints': lista de keypoints por frame (ya procesados)
                       - 'depth_frames': lista de frames de profundidad  
                       
        Returns:
            float: Distancia acumulada en este chunk (metros)
        """
        try:
            keypoints_list = chunk_data.get('keypoints', [])
            depth_frames = chunk_data.get('depth_frames', [])
            
            if not all([keypoints_list, depth_frames]):
                logger.error("El chunk de datos no contiene todas las claves requeridas (keypoints, depth_frames)")
                return 0.0
            
            return self.process_chunk(keypoints_list, depth_frames)
            
        except Exception as e:
            logger.error(f"Error procesando chunk desde datos del servidor: {e}")
            return 0.0

    # Utilidades
    def last_point(self) -> Optional[np.ndarray]:
        return self.trajectory_m[-1] if self.trajectory_m else None

    def stats(self) -> dict:
        return {
            'points': len(self.trajectory_m),
            'total_distance_m': float(self.total_distance_m),
            'last_point': self.last_point().tolist() if self.last_point() is not None else None,
            'camera_intrinsics': self.camera_intrinsics,
            'region': self.region,
            'alpha': self.alpha,
            'min_conf': self.min_conf,
            'min_valid_percent': self.min_valid_percent,
            'max_jump_m': self.max_jump_m,
        }
