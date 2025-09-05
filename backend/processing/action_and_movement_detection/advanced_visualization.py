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
    
    # Configuración de texto moderna
    font = cv2.FONT_HERSHEY_DUPLEX
    font_title = cv2.FONT_HERSHEY_TRIPLEX
    font_scale_title = 0.8
    font_scale_large = 0.7
    font_scale_medium = 0.6
    font_scale_small = 0.5
    thickness_bold = 2
    thickness_normal = 1
    
    # Paleta de colores moderna y profesional
    bg_dark = (20, 25, 30)        # Azul oscuro elegante
    bg_semi = (40, 50, 60)        # Azul medio
    accent_blue = (255, 193, 7)    # Amarillo dorado
    accent_green = (76, 175, 80)   # Verde moderno
    accent_orange = (255, 152, 0)  # Naranja vibrante
    accent_red = (244, 67, 54)     # Rojo moderno
    text_white = (255, 255, 255)   # Blanco puro
    text_light = (220, 220, 220)   # Gris claro
    text_medium = (180, 180, 180)  # Gris medio
    
    # Crear overlay con degradado
    overlay = output_frame.copy()
    
    # ===== HEADER PRINCIPAL CON DEGRADADO =====
    header_height = 80
    # Crear degradado para el header
    for i in range(header_height):
        alpha = 1.0 - (i / header_height) * 0.3  # De opaco a semi-transparente
        color_intensity = int(bg_dark[0] * alpha), int(bg_dark[1] * alpha), int(bg_dark[2] * alpha)
        cv2.line(overlay, (0, i), (width, i), color_intensity, 1)
    
    # Línea accent en la parte superior
    cv2.rectangle(overlay, (0, 0), (width, 4), accent_blue, -1)
    
    # Título principal con sombra
    title_text = f"🎯 ANÁLISIS DE MOVIMIENTO | CÁMARA {camera_id}"
    _draw_text_with_shadow(overlay, title_text, (20, 35), font_title, font_scale_title, text_white, thickness_bold)
    
    # Información del frame con iconos
    frame_info = f"📹 Chunk {chunk_id} • Frame {frame_number:04d}"
    _draw_text_with_shadow(overlay, frame_info, (20, 60), font, font_scale_medium, text_light, thickness_normal)
    
    # ===== PANEL LATERAL DERECHO MODERNO =====
    panel_width = 320
    panel_x = width - panel_width - 15
    panel_y = header_height + 15
    
    # Panel principal con bordes redondeados (simulados)
    _draw_rounded_rect(overlay, (panel_x, panel_y), (panel_width, 350), bg_semi, 15)
    
    # ===== SECCIÓN DE POSTURA =====
    section_y = panel_y + 25
    
    # Título de sección
    _draw_text_with_shadow(overlay, "🏃 DETECCIÓN DE POSTURA", (panel_x + 15, section_y), font, font_scale_medium, accent_green, thickness_bold)
    
    section_y += 35
    if action_detection_result:
        posture = action_detection_result.get('posture', 'indeterminado')
        confidence = action_detection_result.get('confidence', 0.0)
        
        # Status con color según confianza
        status_color = accent_green if confidence > 0.7 else accent_orange if confidence > 0.4 else accent_red
        
        _draw_text_with_shadow(overlay, f"Estado: {posture.upper()}", (panel_x + 15, section_y), font, font_scale_large, status_color, thickness_bold)
        
        # Barra de confianza
        section_y += 25
        _draw_progress_bar(overlay, (panel_x + 15, section_y), 280, 12, confidence, accent_green, bg_dark)
        _draw_text_with_shadow(overlay, f"Confianza: {confidence:.1%}", (panel_x + 15, section_y + 25), font, font_scale_small, text_light, thickness_normal)
        
        # Detalles adicionales
        section_y += 40
        if 'details' in action_detection_result:
            details = action_detection_result['details']
            if 'hip_knee_angle' in details:
                angle = details['hip_knee_angle']
                _draw_text_with_shadow(overlay, f"🦵 Ángulo Cadera-Rodilla: {angle:.1f}°", (panel_x + 15, section_y), font, font_scale_small, text_medium, thickness_normal)
    else:
        _draw_text_with_shadow(overlay, "❌ Sin datos de postura", (panel_x + 15, section_y), font, font_scale_medium, accent_red, thickness_normal)
    
    # ===== SECCIÓN DE TRACKING 3D =====
    section_y += 60
    _draw_text_with_shadow(overlay, "📍 TRACKING DE MARCHA 3D", (panel_x + 15, section_y), font, font_scale_medium, accent_blue, thickness_bold)
    
    section_y += 35
    if gait_tracking_result:
        if 'point_3d' in gait_tracking_result and gait_tracking_result['point_3d'] is not None:
            point_3d = gait_tracking_result['point_3d']
            
            # Coordenadas 3D con formato mejorado
            _draw_text_with_shadow(overlay, f"📐 Posición 3D:", (panel_x + 15, section_y), font, font_scale_small, text_light, thickness_normal)
            section_y += 20
            _draw_text_with_shadow(overlay, f"   X: {point_3d[0]:+.3f}m", (panel_x + 25, section_y), font, font_scale_small, text_medium, thickness_normal)
            section_y += 18
            _draw_text_with_shadow(overlay, f"   Y: {point_3d[1]:+.3f}m", (panel_x + 25, section_y), font, font_scale_small, text_medium, thickness_normal)
            section_y += 18
            _draw_text_with_shadow(overlay, f"   Z: {point_3d[2]:+.3f}m", (panel_x + 25, section_y), font, font_scale_small, text_medium, thickness_normal)
        
        if 'total_distance' in gait_tracking_result:
            total_dist = gait_tracking_result['total_distance']
            section_y += 30
            
            # Distancia total destacada
            _draw_text_with_shadow(overlay, f"🎯 DISTANCIA TOTAL", (panel_x + 15, section_y), font, font_scale_small, text_light, thickness_normal)
            section_y += 25
            _draw_text_with_shadow(overlay, f"{total_dist:.3f} metros", (panel_x + 15, section_y), font, font_scale_large, accent_orange, thickness_bold)
    else:
        _draw_text_with_shadow(overlay, "❌ Sin datos de tracking", (panel_x + 15, section_y), font, font_scale_medium, accent_red, thickness_normal)
    
    # ===== INDICADOR DE CENTRO DE CADERA MEJORADO =====
    if gait_tracking_result and 'point_3d' in gait_tracking_result and gait_tracking_result['point_3d'] is not None:
        # Dibujar indicador moderno en el centro de la cadera
        center_x = width // 2
        center_y = height // 2 + 50
        
        # Círculo principal con glow effect
        cv2.circle(overlay, (center_x, center_y), 20, accent_orange, 2)
        cv2.circle(overlay, (center_x, center_y), 15, accent_orange, 1)
        cv2.circle(overlay, (center_x, center_y), 3, accent_orange, -1)
        
        # Cruz direccional moderna
        cross_size = 12
        cv2.line(overlay, (center_x - cross_size, center_y), (center_x + cross_size, center_y), accent_orange, 3)
        cv2.line(overlay, (center_x, center_y - cross_size), (center_x, center_y + cross_size), accent_orange, 3)
    
    # ===== FOOTER CON TIMESTAMP =====
    footer_y = height - 25
    _draw_text_with_shadow(overlay, f"⏱️  Análisis en tiempo real • Sistema de tracking avanzado", (20, footer_y), font, font_scale_small, text_medium, thickness_normal)
    
    # Combinar overlay con el frame original con transparencia
    alpha = 0.85
    result_frame = output_frame.copy()
    cv2.addWeighted(overlay, alpha, result_frame, 1 - alpha, 0, result_frame)
    
    return result_frame

def _draw_text_with_shadow(img, text, position, font, scale, color, thickness):
    """Dibuja texto con sombra para mejor legibilidad"""
    x, y = position
    # Sombra
    cv2.putText(img, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 1)
    # Texto principal
    cv2.putText(img, text, (x, y), font, scale, color, thickness)

def _draw_rounded_rect(img, top_left, size, color, radius):
    """Simula un rectángulo con bordes redondeados"""
    x, y = top_left
    w, h = size
    
    # Rectángulo principal
    cv2.rectangle(img, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(img, (x, y + radius), (x + w, y + h - radius), color, -1)
    
    # Círculos en las esquinas
    cv2.circle(img, (x + radius, y + radius), radius, color, -1)
    cv2.circle(img, (x + w - radius, y + radius), radius, color, -1)
    cv2.circle(img, (x + radius, y + h - radius), radius, color, -1)
    cv2.circle(img, (x + w - radius, y + h - radius), radius, color, -1)

def _draw_progress_bar(img, position, width, height, progress, color_fill, color_bg):
    """Dibuja una barra de progreso moderna"""
    x, y = position
    
    # Fondo de la barra
    cv2.rectangle(img, (x, y), (x + width, y + height), color_bg, -1)
    
    # Progreso
    fill_width = int(width * max(0, min(1, progress)))
    if fill_width > 0:
        cv2.rectangle(img, (x, y), (x + fill_width, y + height), color_fill, -1)
    
    # Borde
    cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 1)

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
                action_result = manual_action_detector.classify_posture([], lat_kp)
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
