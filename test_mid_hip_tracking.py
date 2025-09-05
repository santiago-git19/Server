"""
Test básico del trackeo del mid_hip con gait_3d_tracker
"""

import sys
import os
import numpy as np
from pathlib import Path

# Configurar path
server_path = Path(__file__).parent
sys.path.append(str(server_path))

def test_gait_tracker_basic():
    """Test básico del gait tracker"""
    
    try:
        from backend.processing.action_and_movement_detection.gait_3d_tracker import Gait3DTracker
        from backend.processing.action_and_movement_detection.advanced_visualization import convert_keypoints_to_gait_format
        print("✅ Importación exitosa de módulos")
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        return
    
    # Parámetros de prueba para Orbbec Gemini 335Le
    intrinsics = {
        'fx': 570.3, 
        'fy': 570.3, 
        'cx': 320.0, 
        'cy': 240.0
    }
    
    # Crear tracker
    tracker = Gait3DTracker(camera_intrinsics=intrinsics)
    print("✅ Gait tracker inicializado")
    
    # Simular keypoints en movimiento (formato TRT: lista de tuplas)
    def create_moving_keypoints(frame_idx, total_frames=10):
        # Simular movimiento horizontal del cuerpo
        movement_x = (frame_idx / total_frames) * 100  # Movimiento de 100 pixels
        base_x = 320 + movement_x
        base_y = 240
        
        keypoints = []
        
        # Añadir solo las caderas (lo que necesita gait tracker)
        # Left hip (part_id = 11)
        keypoints.append((base_x - 30, base_y + 50, 0.9, 11))
        # Right hip (part_id = 12) 
        keypoints.append((base_x + 30, base_y + 50, 0.9, 12))
        
        return keypoints
    
    # Simular frames de profundidad
    def create_depth_frame(frame_idx, total_frames=10):
        # Frame de profundidad de 640x480
        depth_frame = np.full((480, 640), 1500, dtype=np.uint16)  # 1.5 metros de distancia base
        
        # Simular persona moviéndose (área con profundidad variable)
        movement_x = int((frame_idx / total_frames) * 100)
        start_x = max(0, 290 + movement_x)
        end_x = min(640, 350 + movement_x)
        
        # Crear región de persona con profundidad ligeramente diferente
        depth_frame[190:290, start_x:end_x] = 1480 + frame_idx * 2  # Variar ligeramente la profundidad
        
        return depth_frame
    
    print("\n🎯 Probando trackeo de mid_hip...")
    
    total_frames = 10
    for frame_idx in range(total_frames):
        # Generar datos de prueba
        keypoints = create_moving_keypoints(frame_idx, total_frames)
        depth_frame = create_depth_frame(frame_idx, total_frames)
        
        # Procesar frame con tracker
        point_3d = tracker.update(keypoints, depth_frame)
        
        if point_3d is not None:
            print(f"Frame {frame_idx:2d}: 3D Point = ({point_3d[0]:6.3f}, {point_3d[1]:6.3f}, {point_3d[2]:6.3f}) m")
            print(f"            Distancia total = {tracker.total_distance_m:.3f} m")
        else:
            print(f"Frame {frame_idx:2d}: ❌ No se pudo obtener punto 3D")
    
    print(f"\n📊 Estadísticas finales:")
    print(f"   Puntos procesados: {len(tracker.trajectory_m)}")
    print(f"   Distancia total: {tracker.total_distance_m:.3f} metros")
    print(f"   Último punto: {tracker.last_point()}")
    
    # Test de conversión de formatos
    print(f"\n🔄 Probando conversión de formatos...")
    
    # Formato numpy (17, 3)
    numpy_keypoints = np.zeros((17, 3))
    numpy_keypoints[11] = [290, 290, 0.9]  # left_hip
    numpy_keypoints[12] = [350, 290, 0.9]  # right_hip
    
    converted = convert_keypoints_to_gait_format(numpy_keypoints)
    if converted:
        print("✅ Conversión de numpy array exitosa")
        # Verificar que left_hip y right_hip están presentes
        left_hip_found = any(kp[3] == 11 for kp in converted)
        right_hip_found = any(kp[3] == 12 for kp in converted)
        print(f"   Left hip encontrado: {left_hip_found}")
        print(f"   Right hip encontrado: {right_hip_found}")
    else:
        print("❌ Error en conversión de numpy array")
    
    # Formato lista (ya correcto)
    list_keypoints = [(290.0, 290.0, 0.9, 11), (350.0, 290.0, 0.9, 12)]
    converted_list = convert_keypoints_to_gait_format(list_keypoints)
    if converted_list == list_keypoints:
        print("✅ Conversión de lista (sin cambios) exitosa")
    else:
        print("❌ Error en conversión de lista")

def test_mid_hip_extraction():
    """Test específico de extracción del mid_hip"""
    
    try:
        from backend.processing.action_and_movement_detection.gait_3d_tracker import Gait3DTracker
        print("✅ Módulo gait_3d_tracker importado")
    except ImportError as e:
        print(f"❌ Error importando gait_3d_tracker: {e}")
        return
    
    print("\n🎯 Probando extracción de mid_hip...")
    
    # Test con keypoints válidos
    valid_keypoints = [
        (100.0, 200.0, 0.8, 11),  # left_hip con confianza alta
        (140.0, 200.0, 0.9, 12),  # right_hip con confianza alta  
        (120.0, 150.0, 0.7, 5),   # otro keypoint para ruido
    ]
    
    hip_center = Gait3DTracker._extract_hip_center_2d(valid_keypoints, min_conf=0.5)
    if hip_center:
        x, y, conf = hip_center
        expected_x = int((100 + 140) / 2)  # 120
        expected_y = int((200 + 200) / 2)  # 200
        print(f"✅ Mid-hip extraído: ({x}, {y}) conf={conf:.2f}")
        print(f"   Esperado: ({expected_x}, {expected_y})")
        
        if x == expected_x and y == expected_y:
            print("✅ Coordenadas correctas")
        else:
            print("❌ Coordenadas incorrectas")
    else:
        print("❌ No se pudo extraer mid_hip")
    
    # Test con confianza baja
    low_conf_keypoints = [
        (100.0, 200.0, 0.2, 11),  # left_hip con confianza baja
        (140.0, 200.0, 0.3, 12),  # right_hip con confianza baja
    ]
    
    hip_center_low = Gait3DTracker._extract_hip_center_2d(low_conf_keypoints, min_conf=0.5)
    if hip_center_low is None:
        print("✅ Correctamente rechazado por confianza baja")
    else:
        print("❌ Debería haber rechazado por confianza baja")

if __name__ == "__main__":
    print("🧪 Test del trackeo de mid_hip\n")
    
    test_mid_hip_extraction()
    test_gait_tracker_basic()
    
    print("\n✅ Tests completados")
