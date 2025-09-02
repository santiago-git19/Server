import sys
import os
import logging
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

# Añadir el directorio Codigo al path para importar trt_pose_proc
codigo_path = Path(__file__).parent.parent.parent.parent.parent / "Codigo" / "Automatizacion"
sys.path.insert(0, str(codigo_path))

try:
    from utils.pose_detection.trt_pose_proc import TRTPoseProcessor
except ImportError as e:
    TRTPoseProcessor = None
    import_error = str(e)

logger = logging.getLogger(__name__)


class TrtDetector:
    """
    Wrapper para el detector TRT que implementa la interfaz esperada por el coordinador
    """
    
    def __init__(self):
        """Inicializar detector TRT"""
        self.model_name = "trt_pose"
        self.is_initialized = False
        self.processor: Optional[TRTPoseProcessor] = None
        
        # Rutas por defecto - pueden ajustarse según configuración
        self.model_path = codigo_path / "models" / "pose_landmark_lite_fp16.engine"
        self.topology_path = codigo_path / "models" / "human_pose.json"
        
        # Verificar si las rutas alternativas existen
        alt_model_path = codigo_path / "models" / "densenet121_baseline_att_256x256_B_epoch_160.pth"
        if not self.model_path.exists() and alt_model_path.exists():
            self.model_path = alt_model_path
            logger.info(f"Usando modelo alternativo: {self.model_path}")
    
    def initialize(self) -> bool:
        """
        Inicializar el detector TRT
        
        Returns:
            True si la inicialización fue exitosa
        """
        try:
            if TRTPoseProcessor is None:
                logger.error(f"No se pudo importar TRTPoseProcessor: {import_error}")
                return False
            
            # Verificar archivos necesarios
            if not self.model_path.exists():
                logger.error(f"Modelo TRT no encontrado: {self.model_path}")
                return False
                
            if not self.topology_path.exists():
                logger.error(f"Archivo de topología no encontrado: {self.topology_path}")
                return False
            
            logger.info(f"Inicializando TRT detector con modelo: {self.model_path}")
            
            # Crear instancia del procesador
            self.processor = TRTPoseProcessor(
                model_path=str(self.model_path),
                topology_path=str(self.topology_path),
                use_tensorrt=True  # Intentar TensorRT primero
            )
            
            self.is_initialized = True
            logger.info("TRT detector inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando TRT detector: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_chunk(self, video_path: Path, patient_id: str, session_id: str, 
                     camera_id: int, chunk_id: str) -> bool:
        """
        Procesar un chunk de video con el detector TRT
        
        Args:
            video_path: Ruta al archivo de video
            patient_id: ID del paciente
            session_id: ID de la sesión
            camera_id: ID de la cámara
            chunk_id: ID del chunk
            
        Returns:
            True si el procesamiento fue exitoso
        """
        if not self.is_initialized or self.processor is None:
            logger.error("TRT detector no está inicializado")
            return False
            
        try:
            logger.info(f"Procesando chunk TRT: {video_path} (cam {camera_id}, session {session_id})")
            
            # Abrir video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"No se pudo abrir el video: {video_path}")
                return False
            
            frame_count = 0
            keypoints_data = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar frame con TRT
                keypoints = self.processor.process_frame(frame)
                
                # Almacenar resultados
                frame_data = {
                    'frame': frame_count,
                    'keypoints': keypoints,
                    'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS)
                }
                keypoints_data.append(frame_data)
                frame_count += 1
            
            cap.release()
            
            # Guardar resultados
            self._save_results(keypoints_data, patient_id, session_id, camera_id, chunk_id)
            
            logger.info(f"Chunk TRT procesado exitosamente: {frame_count} frames")
            return True
            
        except Exception as e:
            logger.error(f"Error procesando chunk TRT: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_results(self, keypoints_data: list, patient_id: str, session_id: str, 
                     camera_id: int, chunk_id: str):
        """Guardar resultados del procesamiento"""
        try:
            # Crear directorio de salida
            output_dir = Path("results") / patient_id / session_id / f"camera_{camera_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Nombre del archivo de salida
            output_file = output_dir / f"{chunk_id}_trt_pose.json"
            
            # Preparar datos para guardar
            results = {
                'metadata': {
                    'patient_id': patient_id,
                    'session_id': session_id,
                    'camera_id': camera_id,
                    'chunk_id': chunk_id,
                    'detector': self.model_name,
                    'total_frames': len(keypoints_data)
                },
                'keypoints': keypoints_data
            }
            
            # Guardar como JSON
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=self._json_serializer)
            
            logger.info(f"Resultados guardados en: {output_file}")
            
        except Exception as e:
            logger.error(f"Error guardando resultados: {e}")
    
    def _json_serializer(self, obj):
        """Serializador personalizado para JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
