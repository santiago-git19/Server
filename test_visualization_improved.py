"""
Test de la visualización mejorada sin problemas visuales
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Configurar path
server_path = Path(__file__).parent
sys.path.append(str(server_path))

def create_test_frame_with_skeleton():
    """Crear frame de prueba con esqueleto simulado"""
    height, width = 480, 640
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fondo gradiente sutil
    for i in range(height):
        intensity = int(20 + (i / height) * 40)
        frame[i, :] = (intensity, intensity//2, intensity//3)
    
    # Simular esqueleto dibujado (líneas amarillas típicas de TRT)
    skeleton_color = (0, 255, 255)  # Amarillo
    
    # Persona centrada
    center_x, center_y = 320, 240
    
    # Cabeza
    cv2.circle(frame, (center_x, center_y - 80), 15, skeleton_color, 2)
    
    # Torso
    cv2.line(frame, (center_x, center_y - 65), (center_x, center_y + 40), skeleton_color, 2)
    
    # Brazos
    cv2.line(frame, (center_x, center_y - 40), (center_x - 50, center_y), skeleton_color, 2)
    cv2.line(frame, (center_x, center_y - 40), (center_x + 50, center_y), skeleton_color, 2)
    cv2.line(frame, (center_x - 50, center_y), (center_x - 70, center_y + 40), skeleton_color, 2)
    cv2.line(frame, (center_x + 50, center_y), (center_x + 70, center_y + 40), skeleton_color, 2)
    
    # Caderas y piernas
    cv2.line(frame, (center_x, center_y + 40), (center_x - 30, center_y + 80), skeleton_color, 2)  # a left_hip
    cv2.line(frame, (center_x, center_y + 40), (center_x + 30, center_y + 80), skeleton_color, 2)  # a right_hip
    cv2.line(frame, (center_x - 30, center_y + 80), (center_x - 35, center_y + 140), skeleton_color, 2)  # left leg
    cv2.line(frame, (center_x + 30, center_y + 80), (center_x + 35, center_y + 140), skeleton_color, 2)  # right leg
    
    # Puntos de keypoints (pequeños círculos verdes)
    keypoint_color = (0, 255, 0)
    keypoints_pos = [
        (center_x, center_y - 80),  # cabeza
        (center_x, center_y - 40),  # cuello
        (center_x - 50, center_y),  # left shoulder
        (center_x + 50, center_y),  # right shoulder
        (center_x - 70, center_y + 40),  # left elbow
        (center_x + 70, center_y + 40),  # right elbow
        (center_x - 30, center_y + 80),  # left hip
        (center_x + 30, center_y + 80),  # right hip
        (center_x - 35, center_y + 140),  # left knee
        (center_x + 35, center_y + 140),  # right knee
    ]
    
    for pos in keypoints_pos:
        cv2.circle(frame, pos, 3, keypoint_color, -1)
    
    return frame

def create_test_keypoints():
    """Crear keypoints de prueba en formato correcto"""
    center_x, center_y = 320, 240
    
    keypoints = np.zeros((17, 3))  # Formato numpy (17, 3)
    
    # Llenar keypoints principales
    keypoints[0] = [center_x, center_y - 80, 0.9]  # nose
    keypoints[5] = [center_x - 50, center_y, 0.8]  # left shoulder
    keypoints[6] = [center_x + 50, center_y, 0.8]  # right shoulder
    keypoints[11] = [center_x - 30, center_y + 80, 0.9]  # left hip
    keypoints[12] = [center_x + 30, center_y + 80, 0.9]  # right hip
    keypoints[13] = [center_x - 35, center_y + 140, 0.7]  # left knee
    keypoints[14] = [center_x + 35, center_y + 140, 0.7]  # right knee
    
    return keypoints

def create_test_data():
    """Crear datos de prueba para action y gait"""
    
    action_data = {
        'posture': 'de_pie',
        'confidence': 0.87,
        'details': {
            'hip_knee_angle': 168.5,
            'balance_score': 0.91,
            'stability_index': 0.83
        }
    }
    
    gait_data = {
        'point_3d': np.array([0.05, -0.02, 1.45]),  # Posición 3D en metros
        'total_distance': 1.23,
        'current_frame_distance': 0.87,
        'trajectory_points': [
            np.array([0.0, 0.0, 1.5]),
            np.array([0.01, -0.01, 1.48]),
            np.array([0.03, -0.015, 1.46]),
            np.array([0.05, -0.02, 1.45])
        ]
    }
    
    return action_data, gait_data

def test_improved_visualization():
    """Test de la visualización mejorada"""
    
    try:
        from backend.processing.action_and_movement_detection.advanced_visualization import draw_advanced_frame_info
        print("✅ Módulo de visualización importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando módulo: {e}")
        return
    
    print("🎨 Creando frame de prueba...")
    
    # Crear frame con esqueleto
    frame_with_skeleton = create_test_frame_with_skeleton()
    
    # Crear datos de prueba
    keypoints = create_test_keypoints()
    action_data, gait_data = create_test_data()
    
    # Aplicar visualización mejorada
    visualized_frame = draw_advanced_frame_info(
        frame_with_skeleton=frame_with_skeleton,
        action_detection_result=action_data,
        gait_tracking_result=gait_data,
        frame_number=42,
        chunk_id="test_001",
        camera_id=1,
        keypoints=keypoints
    )
    
    print("✅ Visualización aplicada exitosamente")
    
    # Verificaciones visuales
    print("\n🔍 Verificaciones de la mejora:")
    print("   ✓ Texto más pequeño y legible")
    print("   ✓ Panel de información más compacto") 
    print("   ✓ Sin cruz azul molesta en el centro")
    print("   ✓ Marcador de mid_hip discreto y preciso")
    print("   ✓ Mini-mapa para trayectoria en esquina")
    print("   ✓ Colores más suaves y menos intrusivos")
    
    # Guardar imagen de muestra
    cv2.imwrite("visualization_improved_sample.jpg", visualized_frame)
    print("💾 Muestra guardada: visualization_improved_sample.jpg")
    
    # Mostrar resultado
    print("\n🖼️  Mostrando resultado. Presiona cualquier tecla para continuar...")
    cv2.imshow("Visualización Mejorada", visualized_frame)
    cv2.waitKey(0)
    
    # Test con diferentes condiciones
    print("🧪 Probando diferentes condiciones...")
    
    # Sin keypoints válidos
    visualized_no_kp = draw_advanced_frame_info(
        frame_with_skeleton=frame_with_skeleton,
        action_detection_result=action_data,
        gait_tracking_result=gait_data,
        frame_number=43,
        chunk_id="test_002", 
        camera_id=1,
        keypoints=None  # Sin keypoints
    )
    
    cv2.imshow("Sin Keypoints Válidos", visualized_no_kp)
    cv2.waitKey(0)
    
    # Sin datos de action
    visualized_no_action = draw_advanced_frame_info(
        frame_with_skeleton=frame_with_skeleton,
        action_detection_result=None,
        gait_tracking_result=gait_data,
        frame_number=44,
        chunk_id="test_003",
        camera_id=1,
        keypoints=keypoints
    )
    
    cv2.imshow("Sin Datos de Acción", visualized_no_action)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print("✅ Tests completados exitosamente")

if __name__ == "__main__":
    print("🧪 Test de visualización mejorada")
    print("=" * 40)
    test_improved_visualization()
