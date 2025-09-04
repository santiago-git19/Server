"""
Módulo de visualización avanzada que combina:
- Esqueleto dibujado por TRT detector
- Información de manual_action_detector 
- Información de gait_3d_tracker
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

def draw_advanced_frame_info(
    frame_with_skeleton: np.ndarray,
    action_detection_result: Optional[Dict[str, Any]] = None,
    gait_tracking_result: Optional[Dict[str, Any]] = None,
    frame_number: int = 0,
    chunk_id: str = "",
    camera_id: int = 0
) -> np.ndarray:
    """
    Dibuja información avanzada sobre un frame que ya tiene el esqueleto dibujado.
    
    Args:
        frame_with_skeleton: Frame con el esqueleto ya dibujado por TRT detector
        action_detection_result: Resultado de manual_action_detector.classify_posture()
        gait_tracking_result: Resultado de gait_3d_tracker (punto 3D, distancia, etc.)
        frame_number: Número del frame en el chunk
        chunk_id: ID del chunk
        camera_id: ID de la cámara
        
    Returns:
        Frame con toda la información visualizada
    """
    # Crear una copia para no modificar el original
    output_frame = frame_with_skeleton.copy()
    height, width = output_frame.shape[:2]
    
    # Configuración de texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Colores
    bg_color = (0, 0, 0)  # Negro para fondo
    text_color = (255, 255, 255)  # Blanco para texto
    posture_color = (0, 255, 0)  # Verde para postura
    distance_color = (255, 255, 0)  # Amarillo para distancia
    error_color = (0, 0, 255)  # Rojo para errores
    
    # Crear overlay semi-transparente para texto
    overlay = output_frame.copy()
    
    # Panel de información en la parte superior
    panel_height = 120
    cv2.rectangle(overlay, (0, 0), (width, panel_height), bg_color, -1)
    
    # Información básica del frame
    y_offset = 25
    cv2.putText(overlay, f"Camera {camera_id} | Chunk {chunk_id} | Frame {frame_number}", 
                (10, y_offset), font, font_scale, text_color, thickness)
    
    # Información de detección de acciones
    y_offset += 30
    if action_detection_result:
        posture = action_detection_result.get('posture', 'indeterminado')
        confidence = action_detection_result.get('confidence', 0.0)
        
        posture_text = f"Posture: {posture} ({confidence:.2f})"
        cv2.putText(overlay, posture_text, (10, y_offset), font, font_scale, posture_color, thickness)
        
        # Mostrar detalles adicionales si están disponibles
        if 'details' in action_detection_result:
            details = action_detection_result['details']
            if 'hip_knee_angle' in details:
                angle_text = f"Hip-Knee Angle: {details['hip_knee_angle']:.1f}°"
                cv2.putText(overlay, angle_text, (300, y_offset), font, font_scale-0.1, text_color, 1)
    else:
        cv2.putText(overlay, "Posture: No data", (10, y_offset), font, font_scale, error_color, thickness)
    
    # Información de tracking de marcha
    y_offset += 30
    if gait_tracking_result:
        if 'point_3d' in gait_tracking_result and gait_tracking_result['point_3d'] is not None:
            point_3d = gait_tracking_result['point_3d']
            distance_text = f"3D Position: ({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f})m"
            cv2.putText(overlay, distance_text, (10, y_offset), font, font_scale-0.1, distance_color, thickness)
        
        if 'total_distance' in gait_tracking_result:
            total_dist = gait_tracking_result['total_distance']
            dist_text = f"Total Distance: {total_dist:.3f}m"
            cv2.putText(overlay, dist_text, (400, y_offset), font, font_scale, distance_color, thickness)
    else:
        cv2.putText(overlay, "Gait Tracking: No data", (10, y_offset), font, font_scale, error_color, thickness)
    
    # Panel lateral para información detallada
    if action_detection_result and 'details' in action_detection_result:
        details = action_detection_result['details']
        panel_x = width - 250
        panel_y = panel_height + 10
        panel_width = 240
        panel_detail_height = 150
        
        # Fondo para panel de detalles
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_detail_height), bg_color, -1)
        
        detail_y = panel_y + 20
        cv2.putText(overlay, "Detection Details:", (panel_x + 5, detail_y), font, font_scale-0.1, text_color, 1)
        
        detail_y += 20
        if 'frontal_result' in action_detection_result:
            frontal = action_detection_result['frontal_result']
            if 'posture' in frontal:
                cv2.putText(overlay, f"Frontal: {frontal['posture']}", (panel_x + 5, detail_y), font, font_scale-0.2, text_color, 1)
                detail_y += 15
        
        if 'lateral_result' in action_detection_result:
            lateral = action_detection_result['lateral_result']
            if 'posture' in lateral:
                cv2.putText(overlay, f"Lateral: {lateral['posture']}", (panel_x + 5, detail_y), font, font_scale-0.2, text_color, 1)
                detail_y += 15
        
        # Mostrar métricas adicionales
        for key, value in details.items():
            if isinstance(value, (int, float)) and key != 'hip_knee_angle':
                if detail_y < panel_y + panel_detail_height - 15:
                    text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
                    cv2.putText(overlay, text, (panel_x + 5, detail_y), font, font_scale-0.3, text_color, 1)
                    detail_y += 12
    
    # Combinar overlay con transparencia
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)
    
    # Dibujar indicadores visuales adicionales
    if gait_tracking_result and 'point_3d' in gait_tracking_result and gait_tracking_result['point_3d'] is not None:
        # Dibujar una cruz en el centro de la cadera proyectado
        point_3d = gait_tracking_result['point_3d']
        # Proyectar punto 3D de vuelta a 2D para visualización (aproximación)
        # Esto es una aproximación simple, idealmente usarías los parámetros intrínsecos reales
        center_x = width // 2
        center_y = height // 2 + 50  # Offset hacia abajo para cadera
        
        cross_size = 10
        cv2.line(output_frame, (center_x - cross_size, center_y), (center_x + cross_size, center_y), distance_color, 2)
        cv2.line(output_frame, (center_x, center_y - cross_size), (center_x, center_y + cross_size), distance_color, 2)
        cv2.circle(output_frame, (center_x, center_y), cross_size + 5, distance_color, 2)
    
    return output_frame

def save_annotated_chunk_video(
    annotated_frames: List[np.ndarray],
    output_path: Path,
    fps: float = 30.0
) -> bool:
    """
    Guarda una lista de frames anotados como un video.
    
    Args:
        annotated_frames: Lista de frames con anotaciones
        output_path: Ruta donde guardar el video
        fps: Frames por segundo del video de salida
        
    Returns:
        True si se guardó correctamente
    """
    try:
        if not annotated_frames:
            logger.error("No hay frames para guardar")
            return False
        
        # Crear directorio si no existe
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Obtener dimensiones del primer frame
        height, width = annotated_frames[0].shape[:2]
        
        # Configurar codec y writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"No se pudo abrir el writer de video: {output_path}")
            return False
        
        # Escribir frames
        for frame in annotated_frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Video anotado guardado: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error guardando video anotado: {e}")
        return False

def process_chunk_with_advanced_visualization(
    trt_detector,
    manual_action_detector,
    gait_tracker,
    video_path: Path,
    depth_frames: List[np.ndarray],
    camera_id: int,
    chunk_id: str,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Procesa un chunk completo con visualización avanzada.
    
    Args:
        trt_detector: Instancia del detector TRT
        manual_action_detector: Instancia del detector de acciones manuales
        gait_tracker: Instancia del tracker de marcha
        video_path: Ruta del video del chunk
        depth_frames: Lista de frames de profundidad
        camera_id: ID de la cámara
        chunk_id: ID del chunk
        output_dir: Directorio donde guardar resultados
        
    Returns:
        Diccionario con resultados del procesamiento
    """
    try:
        logger.info(f"Procesando chunk {chunk_id} con visualización avanzada")
        
        # Procesar chunk para obtener keypoints y frames visualizados
        keypoints_list, visualized_frames = trt_detector.process_and_visualize_chunk(video_path)
        
        if not keypoints_list or not visualized_frames:
            logger.error("No se pudieron obtener keypoints o frames visualizados")
            return {'success': False, 'error': 'No keypoints or visualized frames'}
        
        # Procesar con gait tracker
        gait_distance = gait_tracker.process_chunk(keypoints_list, depth_frames)
        
        # Para manual action detector, necesitamos separar keypoints por cámara
        # Asumiendo que tenemos al menos 2 cámaras (frontal y lateral)
        frontal_keypoints = keypoints_list if camera_id == 0 else []
        lateral_keypoints = keypoints_list if camera_id == 1 else []
        
        # Procesar detección de acciones (simplificado para un solo set de keypoints)
        action_results = []
        if frontal_keypoints and lateral_keypoints:
            for frame_idx, (front_kp, lat_kp) in enumerate(zip(frontal_keypoints, lateral_keypoints)):
                action_result = manual_action_detector.classify_posture(front_kp, lat_kp)
                action_results.append(action_result)
        
        # Crear frames con visualización avanzada
        annotated_frames = []
        for frame_idx, vis_frame in enumerate(visualized_frames):
            # Obtener resultados para este frame
            action_result = action_results[frame_idx] if frame_idx < len(action_results) else None
            
            # Información de gait tracking para este frame
            gait_info = {
                'total_distance': gait_tracker.total_distance_m,
                'point_3d': gait_tracker.last_point()
            }
            
            # Dibujar información avanzada
            annotated_frame = draw_advanced_frame_info(
                frame_with_skeleton=vis_frame,
                action_detection_result=action_result,
                gait_tracking_result=gait_info,
                frame_number=frame_idx,
                chunk_id=chunk_id,
                camera_id=camera_id
            )
            
            annotated_frames.append(annotated_frame)
        
        # Guardar video anotado
        output_video_path = output_dir / f"camera_{camera_id}_chunk_{chunk_id}_annotated.mp4"
        save_success = save_annotated_chunk_video(annotated_frames, output_video_path)
        
        # Preparar resultados
        results = {
            'success': True,
            'video_path': str(output_video_path) if save_success else None,
            'gait_distance': gait_distance,
            'total_gait_distance': gait_tracker.total_distance_m,
            'processed_frames': len(annotated_frames),
            'action_results': action_results,
            'gait_trajectory_points': len(gait_tracker.trajectory_m)
        }
        
        logger.info(f"Chunk {chunk_id} procesado exitosamente con visualización avanzada")
        return results
        
    except Exception as e:
        logger.error(f"Error procesando chunk con visualización avanzada: {e}")
        return {'success': False, 'error': str(e)}
