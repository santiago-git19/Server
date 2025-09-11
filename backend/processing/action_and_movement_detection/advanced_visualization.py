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

def convert_keypoints_to_gait_format(keypoints) -> Optional[List[Tuple[float, float, float, int]]]:
    """
    Convierte keypoints de diferentes formatos al formato esperado por gait_3d_tracker.
    
    Args:
        keypoints: Keypoints en formato numpy (17,3) o lista de tuplas (x,y,conf,part_id)
        
    Returns:
        Lista de tuplas (x, y, conf, part_id) o None si no se puede convertir
    """
    if keypoints is None:
        return None
    
    # Si ya está en formato de lista de tuplas (x, y, conf, part_id)
    if isinstance(keypoints, list) and len(keypoints) > 0:
        # Verificar si es una lista de tuplas con 4 elementos
        if isinstance(keypoints[0], tuple) and len(keypoints[0]) == 4:
            return keypoints
        # Verificar si es una lista de listas con 4 elementos
        elif isinstance(keypoints[0], list) and len(keypoints[0]) == 4:
            return [(float(x), float(y), float(conf), int(part_id)) for x, y, conf, part_id in keypoints]
        # Verificar si es una lista de tuplas con 3 elementos (sin part_id)
        elif isinstance(keypoints[0], (tuple, list)) and len(keypoints[0]) == 3:
            return [(float(x), float(y), float(conf), int(i)) for i, (x, y, conf) in enumerate(keypoints)]
    
    # Si está en formato numpy array (17, 3)
    if isinstance(keypoints, np.ndarray) and keypoints.shape == (17, 3):
        converted = []
        for part_id in range(17):
            x, y, conf = keypoints[part_id]
            converted.append((float(x), float(y), float(conf), int(part_id)))
        return converted
    
    # Si está en formato numpy array (N, 3) pero no es 17
    if isinstance(keypoints, np.ndarray) and len(keypoints.shape) == 2 and keypoints.shape[1] == 3:
        converted = []
        for part_id in range(keypoints.shape[0]):
            x, y, conf = keypoints[part_id]
            converted.append((float(x), float(y), float(conf), int(part_id)))
        return converted
    
    # Si es una lista pero no pudimos procesarla arriba
    if isinstance(keypoints, list):
        logger.warning(f"Lista de keypoints con formato no reconocido. Longitud: {len(keypoints)}, "
                      f"Primer elemento: {keypoints[0] if keypoints else 'vacío'}, "
                      f"Tipo primer elemento: {type(keypoints[0]) if keypoints else 'N/A'}")
    else:
        logger.warning(f"Formato de keypoints no reconocido: {type(keypoints)}")
    
    return None

def draw_advanced_frame_info(
    frame_with_skeleton: np.ndarray,
    action_detection_result: Optional[Dict[str, Any]] = None,
    gait_tracking_result: Optional[Dict[str, Any]] = None,
    frame_number: int = 0,
    chunk_id: str = "",
    camera_id: int = 0,
    keypoints: Optional[np.ndarray] = None
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
        keypoints: Array de keypoints (17, 3) para obtener coordenadas reales del mid_hip
        
    Returns:
        Frame con toda la información visualizada
    """
    # Crear una copia para no modificar el original
    output_frame = frame_with_skeleton.copy()
    height, width = output_frame.shape[:2]
    
    # Configuración de texto - reducir tamaños
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4  # Reducido de 0.6
    thickness = 1     # Reducido de 2
    
    # Colores
    bg_color = (0, 0, 0)  # Negro para fondo
    text_color = (255, 255, 255)  # Blanco para texto
    posture_color = (0, 255, 0)  # Verde para postura
    distance_color = (255, 255, 0)  # Amarillo para distancia
    error_color = (0, 0, 255)  # Rojo para errores
    
    # Crear overlay semi-transparente para texto
    overlay = output_frame.copy()
    
    # Panel de información en la parte superior - más compacto
    panel_height = 80  # Reducido de 120
    cv2.rectangle(overlay, (0, 0), (width, panel_height), bg_color, -1)
    
    # Información básica del frame
    y_offset = 18  # Reducido de 25
    cv2.putText(overlay, f"Cam{camera_id} | F{frame_number} | {chunk_id}", 
                (10, y_offset), font, font_scale, text_color, thickness)
    
    # Información de detección de acciones
    y_offset += 20  # Reducido de 30
    if action_detection_result:
        posture = action_detection_result.get('posture', 'indeterminado')
        confidence = action_detection_result.get('confidence', 0.0)
        
        posture_text = f"Postura: {posture} ({confidence:.2f})"
        cv2.putText(overlay, posture_text, (10, y_offset), font, font_scale, posture_color, thickness)
        
        # Mostrar detalles adicionales si están disponibles
        if 'details' in action_detection_result:
            details = action_detection_result['details']
            if 'hip_knee_angle' in details:
                angle_text = f"Angulo: {details['hip_knee_angle']:.1f}°"
                cv2.putText(overlay, angle_text, (250, y_offset), font, font_scale*0.8, text_color, 1)
    else:
        cv2.putText(overlay, "Postura: Sin datos", (10, y_offset), font, font_scale, error_color, thickness)
    
    # Información de tracking de marcha
    y_offset += 20  # Reducido de 30
    if gait_tracking_result:        
        # Mostrar distancias de forma más compacta
        if 'total_distance' in gait_tracking_result and 'current_frame_distance' in gait_tracking_result:
            total_dist = gait_tracking_result['total_distance']
            frame_dist = gait_tracking_result['current_frame_distance']
            dist_text = f"Distancia: {frame_dist:.2f}m / {total_dist:.2f}m total"
            cv2.putText(overlay, dist_text, (10, y_offset), font, font_scale, distance_color, thickness)
        elif 'total_distance' in gait_tracking_result:
            total_dist = gait_tracking_result['total_distance']
            dist_text = f"Distancia total: {total_dist:.2f}m"
            cv2.putText(overlay, dist_text, (10, y_offset), font, font_scale, distance_color, thickness)
        
        # Mostrar posición 3D de forma compacta solo si es útil
        if 'point_3d' in gait_tracking_result and gait_tracking_result['point_3d'] is not None:
            point_3d = gait_tracking_result['point_3d']
            pos_text = f"Pos: ({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f})m"
            cv2.putText(overlay, pos_text, (300, y_offset), font, font_scale*0.8, (200, 200, 255), 1)
    else:
        cv2.putText(overlay, "Marcha: Sin datos", (10, y_offset), font, font_scale, error_color, thickness)
    
    # Información adicional compacta en la tercera línea (solo lo más importante)
    y_offset += 20
    if action_detection_result and 'details' in action_detection_result:
        details = action_detection_result['details']
        
        # Mostrar solo métricas clave de forma compacta
        metrics_text = []
        if 'balance_score' in details:
            metrics_text.append(f"Balance: {details['balance_score']:.2f}")
        if 'stability_index' in details:
            metrics_text.append(f"Estab: {details['stability_index']:.2f}")
        
        if metrics_text:
            combined_text = " | ".join(metrics_text)
            cv2.putText(overlay, combined_text, (10, y_offset), font, font_scale*0.8, (200, 200, 200), 1)
    
    # Combinar overlay con transparencia más sutil
    alpha = 0.7  # Reducido de 0.8 para ser menos intrusivo
    cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)
    
    # Dibujar indicadores visuales adicionales usando coordenadas reales del mid_hip
    mid_hip_2d = None
    
    # Extraer coordenadas del mid_hip usando la misma lógica que gait_3d_tracker
    if keypoints is not None:
        # Debug: mostrar información sobre los keypoints recibidos
        logger.debug(f"Keypoints recibidos - Tipo: {type(keypoints)}, "
                    f"Es lista: {isinstance(keypoints, list)}, "
                    f"Es numpy: {isinstance(keypoints, np.ndarray)}")
        if isinstance(keypoints, list) and len(keypoints) > 0:
            logger.debug(f"Primer elemento de la lista: {keypoints[0]}, Tipo: {type(keypoints[0])}")
        elif isinstance(keypoints, np.ndarray):
            logger.debug(f"Shape del array numpy: {keypoints.shape}")
        
        # Detectar formato de keypoints y extraer mid_hip
        left_hip = None
        right_hip = None
        
        # Si keypoints es numpy array (17, 3) - formato típico de TRT Pose
        if isinstance(keypoints, np.ndarray) and keypoints.shape == (17, 3):
            left_hip_data = keypoints[11]  # COCO left_hip
            right_hip_data = keypoints[12]  # COCO right_hip
            
            if left_hip_data[2] > 0.01 and right_hip_data[2] > 0.01:
                left_hip = (float(left_hip_data[0]), float(left_hip_data[1]), float(left_hip_data[2]))
                right_hip = (float(right_hip_data[0]), float(right_hip_data[1]), float(right_hip_data[2]))
        
        # Si keypoints es lista de tuplas (x, y, conf, part_id) - formato del gait_tracker
        elif isinstance(keypoints, list):
            for x, y, conf, part_id in keypoints:
                if conf < 0.01:
                    continue
                if part_id == 11:  # COCO_LEFT_HIP
                    left_hip = (float(x), float(y), float(conf))
                elif part_id == 12:  # COCO_RIGHT_HIP
                    right_hip = (float(x), float(y), float(conf))
        # Calcular mid_hip si tenemos ambas caderas
        print("left_hip: ", left_hip)
        print("right_hip: ", right_hip)
        if left_hip and right_hip:
            mid_hip_x = int((left_hip[0] + right_hip[0]) / 2)
            mid_hip_y = int((left_hip[1] + right_hip[1]) / 2)
            mid_hip_2d = (mid_hip_x, mid_hip_y)
            
            # Dibujar marcador más discreto para mid_hip
            cross_size = 4  # Reducido de 8
            hip_color = (0, 200, 255)  # Color naranja más suave

            # Cruz más pequeña y sutil
            cv2.line(output_frame, (mid_hip_x - cross_size, mid_hip_y), (mid_hip_x + cross_size, mid_hip_y), hip_color, 1)
            cv2.line(output_frame, (mid_hip_x, mid_hip_y - cross_size), (mid_hip_x, mid_hip_y + cross_size), hip_color, 1)
            cv2.circle(output_frame, (mid_hip_x, mid_hip_y), 2, hip_color, -1)  # Punto central pequeño
            
            # Texto más pequeño y menos intrusivo
            min_conf = min(left_hip[2], right_hip[2])
            if min_conf > 0.7:  # Solo mostrar si confianza alta
                conf_text = f"{min_conf:.1f}"
                cv2.putText(output_frame, conf_text, (mid_hip_x + 8, mid_hip_y - 8), font, font_scale*0.6, hip_color, 1)
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
        lateral_keypoints = keypoints_list #if camera_id == 0 else []
        
        # Procesar detección de acciones (simplificado para un solo set de keypoints)
        action_results = []
        if frontal_keypoints and lateral_keypoints:
            for frame_idx, (front_kp, lat_kp) in enumerate(zip(frontal_keypoints, lateral_keypoints)):
                print("----------------------------Lateral Keypoints: " + str(lat_kp) + "----------------------------")
                action_result = manual_action_detector.classify_posture([], lat_kp)
                action_results.append(action_result)
        
        # Crear frames con visualización avanzada
        annotated_frames = []
        
        # Resetear gait tracker para este chunk
        initial_distance = gait_tracker.total_distance_m
        
        for frame_idx, vis_frame in enumerate(visualized_frames):
            # Obtener resultados para este frame
            action_result = action_results[frame_idx] if frame_idx < len(action_results) else None
            keypoints_current = keypoints_list[frame_idx] if frame_idx < len(keypoints_list) else None
            depth_frame_current = depth_frames[frame_idx] if frame_idx < len(depth_frames) else None
            
            # Procesar este frame específico con gait tracker si tenemos datos válidos
            current_3d_point = None
            if keypoints_current is not None and depth_frame_current is not None:
                # Convertir keypoints al formato correcto para gait tracker
                keypoints_gait_format = convert_keypoints_to_gait_format(keypoints_current)
                if keypoints_gait_format is not None:
                    # Actualizar gait tracker con este frame específico
                    current_3d_point = gait_tracker.update(keypoints_gait_format, depth_frame_current)
                else:
                    logger.warning(f"No se pudieron convertir keypoints en frame {frame_idx}")
            
            # Si no pudimos procesar con gait tracker, usar último punto conocido
            if current_3d_point is None:
                current_3d_point = gait_tracker.last_point()
            
            # Calcular distancia acumulada hasta este frame
            frame_distance = gait_tracker.total_distance_m - initial_distance
            
            gait_info = {
                'total_distance': gait_tracker.total_distance_m,
                'point_3d': current_3d_point,
                'current_frame_distance': frame_distance,
                'trajectory_points': gait_tracker.trajectory_m.copy() if gait_tracker.trajectory_m else []
            }
            
            # Dibujar información avanzada con keypoints reales
            annotated_frame = draw_advanced_frame_info(
                frame_with_skeleton=vis_frame,
                action_detection_result=action_result,
                gait_tracking_result=gait_info,
                frame_number=frame_idx,
                chunk_id=chunk_id,
                camera_id=camera_id,
                keypoints=keypoints_current
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
