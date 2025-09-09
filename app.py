from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import threading
import numpy as np
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar configuraciones
from config import server_config, data_config, gpu_config
from backend.processing.coordinator import PoseProcessingCoordinator
from backend.processing.ensemble import EnsembleProcessor

# Importar nuevos módulos para visualización avanzada
from backend.processing.action_and_movement_detection.manual_action_detector import PostureClassifier
from backend.processing.action_and_movement_detection.gait_3d_tracker import Gait3DTracker
from backend.processing.action_and_movement_detection.advanced_visualization import process_chunk_with_advanced_visualization
from backend.processing.detectors.trt_detector import TrtDetector  # Import TrtDetector

# Crear aplicación Flask
app = Flask(__name__)
CORS(app)

# Configuración Flask
app.config['MAX_CONTENT_LENGTH'] = server_config.max_content_length
app.config['UPLOAD_FOLDER'] = server_config.upload_folder

# Inicializar directorios
data_config.ensure_directories()

# Inicializar coordinador de procesamiento
pose_coordinator = PoseProcessingCoordinator()

# Inicializar procesador de ensemble
ensemble_processor = EnsembleProcessor(data_config.base_data_dir)

# Lock para evitar múltiples inicializaciones concurrentes
coordinator_lock = threading.Lock()

# Semáforo para permitir chunks procesándose simultáneamente según configuración GPU
processing_semaphore = threading.Semaphore(gpu_config.max_concurrent_chunks)

# Variable global para sesión actual (puede haber hasta una grabando, pero varias procesando chunks)
current_session = {
    'patient_id': None,
    'session_id': None,
    'is_active': False,
    'cameras_count': 0
}

# Variable para controlar si ya verificamos el chunk 2
chunk_2_verified = False

# Inicializar detectores avanzados
# Parámetros intrínsecos de la cámara (Orbbec Gemini 335Le - valores aproximados)
camera_intrinsics = {
    'fx': 375.0805358886719,
    'fy': 375.0805358886719,
    'cx': 320.6000061035156,
    'cy': 241.5
}

# Instancias globales para análisis avanzado
posture_classifier = PostureClassifier(confidence_threshold=0.01)
gait_tracker = Gait3DTracker(camera_intrinsics=camera_intrinsics)

# Lock para gait tracker (para evitar concurrencia entre chunks)
gait_tracker_lock = threading.Lock()

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de salud del servidor"""
    return jsonify({
        'status': 'healthy',
        'service': 'gait-analysis-server',
        'version': '1.0.0'
    })

@app.route('/api/session/status', methods=['GET'])
def get_session_status():
    """Obtener estado de la sesión actual"""
    return jsonify({
        'session_active': current_session['is_active'],
        'current_session': current_session if current_session['is_active'] else None
    })

@app.route('/api/session/start', methods=['POST'])
def start_session():
    """
    Iniciar nueva sesión de procesamiento
    
    Se reciben datos con el siguiente formato:
    {
        "patient_id": "string",
        "session_id": "string", 
        "cameras_count": int
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        patient_id = data.get('patient_id')
        session_id = data.get('session_id') 
        cameras_count = data.get('cameras_count', 3)
        
        if not patient_id or not session_id:
            return jsonify({'error': 'patient_id and session_id are required'}), 400
        
        # Verificar si ya hay una sesión activa y finalizarla automáticamente
        if current_session['is_active']:
            logger.info(f"Sesión activa detectada, finalizando automáticamente: "
                       f"patient{current_session['patient_id']}/session{current_session['session_id']}")
            
            # Finalizar sesión anterior automáticamente (sin eliminar datos)
            old_patient_id = current_session['patient_id']
            old_session_id = current_session['session_id']
            
            current_session.update({
                'patient_id': None,
                'session_id': None,
                'is_active': False,
                'cameras_count': 0
            })
            
            logger.info(f"Sesión anterior finalizada automáticamente: patient{old_patient_id}/session{old_session_id}")
        
        # Crear directorios para la nueva sesión
        session_dirs = []
        for camera_id in range(cameras_count):
            base_dir = data_config.unprocessed_dir / f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}"
            base_dir.mkdir(parents=True, exist_ok=True)
            session_dirs.append(str(base_dir))
        
        # Activar sesión
        current_session.update({
            'patient_id': patient_id,
            'session_id': session_id,
            'is_active': True,
            'cameras_count': cameras_count
        })
        
        # Registrar sesión en ensemble processor
        ensemble_processor.register_session_start(patient_id, session_id, cameras_count)
        
        # Reiniciar gait tracker para nueva sesión
        with gait_tracker_lock:
            gait_tracker.reset()
            logger.info("Gait tracker reiniciado para nueva sesión")
        
        # Reiniciar flag de verificación de chunk 2. Esto es para cuando las cámaras fallan, que algunas graban chunks y otras no. Si se recibe el primer chunk 2, se verificará que todas las cámaras tengan al menos el chunk 0.
        global chunk_2_verified
        chunk_2_verified = False
        
        logger.info(f"Sesión iniciada - Paciente: {patient_id}, Sesión: {session_id}, Cámaras: {cameras_count}")
        
        return jsonify({
            'status': 'session_started',
            'patient_id': patient_id,
            'session_id': session_id,
            'cameras_count': cameras_count,
            'directories_created': session_dirs
        })
        
    except Exception as e:
        logger.error(f"Error iniciando sesión: {str(e)}")
        return jsonify({'error': f'Failed to start session: {str(e)}'}), 500
    
@app.route('/api/session/cancel', methods=['POST'])
def cancel_session():
    """
    Cancelar sesión y grabación actual y limpiar datos
    """
    try:
        if not current_session['is_active']:
            return jsonify({'error': 'No active session to cancel'}), 400
        
        patient_id = current_session['patient_id']
        session_id = current_session['session_id']
        
        # Detener coordinador si está inicializado
        if pose_coordinator.initialized:
            logger.info("Deteniendo coordinador de procesamiento...")
        
        # Limpiar directorios de la sesión cancelada
        session_path = data_config.unprocessed_dir / f"patient{patient_id}" / f"session{session_id}"
        
        if session_path.exists():
            import shutil
            shutil.rmtree(session_path)
            logger.info(f"Directorio de sesión eliminado: {session_path}")
        
        # También limpiar datos procesados si existen
        processed_paths = [
            data_config.photos_dir / f"patient{patient_id}" / f"session{session_id}",
            data_config.keypoints_2d_dir / f"patient{patient_id}" / f"session{session_id}",
            data_config.keypoints_3d_dir / f"patient{patient_id}" / f"session{session_id}"
        ]
        
        for path in processed_paths:
            if path.exists():
                import shutil
                shutil.rmtree(path)
                logger.info(f"Directorio procesado eliminado: {path}")
        
        # Reiniciar sesión
        logger.info(f"Sesión cancelada - Paciente: {patient_id}, Sesión: {session_id}")
        
        old_session = current_session.copy()
        current_session.update({
            'patient_id': None,
            'session_id': None,
            'is_active': False,
            'cameras_count': 0
        })
        
        return jsonify({
            'status': 'session_cancelled',
            'cancelled_session': {
                'patient_id': old_session['patient_id'],
                'session_id': old_session['session_id']
            }
        })
        
    except Exception as e:
        logger.error(f"Error cancelando sesión: {str(e)}")
        return jsonify({'error': f'Failed to cancel session: {str(e)}'}), 500
    

@app.route('/api/session/end', methods=['POST'])
def end_session():
    """
    Finalizar sesión normalmente (mantener datos para procesamiento)
    """
    try:
        if not current_session['is_active']:
            return jsonify({'error': 'No active session to end'}), 400
        
        patient_id = current_session['patient_id']
        session_id = current_session['session_id']
        
        # Registrar finalización de sesión y obtener max_chunk
        max_chunk = ensemble_processor.register_session_end(patient_id, session_id)
        
        logger.info(f"Sesión finalizada normalmente - Paciente: {patient_id}, Sesión: {session_id}, Max chunk: {max_chunk}")
        
        # El ensemble se procesará automáticamente cuando todas las cámaras completen el chunk final
        if max_chunk >= 0:
            logger.info(f"Esperando que todas las cámaras completen el chunk final {max_chunk} para iniciar ensemble")
        else:
            logger.warning("No se encontraron chunks para procesar en ensemble")
        
        old_session = current_session.copy()
        current_session.update({
            'patient_id': None,
            'session_id': None,
            'is_active': False,
            'cameras_count': 0
        })
        
        return jsonify({
            'status': 'session_ended',
            'ended_session': {
                'patient_id': old_session['patient_id'],
                'session_id': old_session['session_id']
            },
            'message': 'Session ended successfully, data preserved for processing'
        })
        
    except Exception as e:
        logger.error(f"Error finalizando sesión: {str(e)}")
        return jsonify({'error': f'Failed to end session: {str(e)}'}), 500

def _check_camera_chunks_integrity(patient_id: str, session_id: str, cameras_count: int):
    """
    Verificar que todas las cámaras tengan al menos el chunk 0 con color y profundidad
    """
    try:
        session_base = data_config.unprocessed_dir / f"patient{patient_id}" / f"session{session_id}"
        
        for camera_id in range(cameras_count):
            camera_dir = session_base / f"camera{camera_id}"
            chunk_0_color = camera_dir / "0_color.mp4"
            chunk_0_depth = camera_dir / "0_depth.npy"
            
            if not chunk_0_color.exists():
                logger.error(f" FALLO DE CÁMARAS: La cámara {camera_id} NO tiene chunk 0 de color")
                return False
                
            if not chunk_0_depth.exists():
                logger.error(f" FALLO DE CÁMARAS: La cámara {camera_id} NO tiene chunk 0 de profundidad")
                return False
        
        logger.info(f" Verificación de integridad OK: Todas las {cameras_count} cámaras tienen chunk 0 (color y profundidad)")
        return True
        
    except Exception as e:
        logger.error(f"Error verificando integridad de chunks: {e}")
        return False

def _cancel_session_due_to_camera_failure():
    """
    Cancelar sesión debido a fallo de cámaras
    """
    try:
        if not current_session['is_active']:
            return
        
        patient_id = current_session['patient_id']
        session_id = current_session['session_id']
        
        logger.error("CANCELANDO SESIÓN POR FALLO DE CÁMARAS - DESCONECTAR Y CONECTAR EL SWITCH, Y REINICIAR LOS SERVIDORES DE FLASK, VOLVIENDO A ABRIR EL FRONTEND")
        
        # Detener coordinador si está inicializado
        if pose_coordinator.initialized:
            logger.info("Deteniendo coordinador de procesamiento...")
        
        # Limpiar directorios de la sesión cancelada
        session_path = data_config.unprocessed_dir / f"patient{patient_id}" / f"session{session_id}"
        
        if session_path.exists():
            import shutil
            shutil.rmtree(session_path)
            logger.info(f"Directorio de sesión eliminado: {session_path}")
        
        # También limpiar datos procesados si existen
        processed_paths = [
            data_config.photos_dir / f"patient{patient_id}" / f"session{session_id}",
            data_config.keypoints_2d_dir / f"patient{patient_id}" / f"session{session_id}",
            data_config.keypoints_3d_dir / f"patient{patient_id}" / f"session{session_id}"
        ]
        
        for path in processed_paths:
            if path.exists():
                import shutil
                shutil.rmtree(path)
                logger.info(f"Directorio procesado eliminado: {path}")
        
        # Reiniciar sesión
        logger.info(f"Sesión cancelada por fallo de cámaras - Paciente: {patient_id}, Sesión: {session_id}")
        
        current_session.update({
            'patient_id': None,
            'session_id': None,
            'is_active': False,
            'cameras_count': 0
        })
        
        # Reiniciar flag de verificación
        global chunk_2_verified
        chunk_2_verified = False
        
    except Exception as e:
        logger.error(f"Error cancelando sesión por fallo de cámaras: {str(e)}")

@app.route('/api/chunks/receive', methods=['POST'])
def receive_chunk():
    """
    Recibir chunk de video para procesamiento
    
    Form data:
    - file_color: archivo de video de color (.mp4)
    - file_depth: archivo de datos de profundidad (.npy)
    - camera_id: ID de la cámara (0, 1, 2...)
    - chunk_number: número del chunk
    - chunk_id: ID único del chunk
    - patient_id: ID del paciente
    - session_id: ID de la sesión
    """
    try:
        # Verificar sesión activa
        if not current_session['is_active']:
            return jsonify({'error': 'No active session'}), 400
        
        # Verificar archivos
        if 'file_color' not in request.files or 'file_depth' not in request.files:
            print("-----------------++----------------------")
            print(request.files)
            print('file_color' not in request.files)
            print('file_depth' not in request.files)
            return jsonify({'error': 'Both file_color and file_depth are required'}), 400
        
        file_color = request.files['file_color']
        file_depth = request.files['file_depth']
        
        if file_color.filename == '' or file_depth.filename == '':
            print("+++++++++++++++++++++++++++++++++++++++++++")
            print(request.files)
            return jsonify({'error': 'Both file_color and file_depth must be provided'}), 400
        
        # Obtener metadatos
        camera_id = request.form.get('camera_id')
        chunk_number = request.form.get('chunk_number')
        chunk_id = request.form.get('chunk_id')
        
        if camera_id is None or chunk_number is None:
            return jsonify({'error': 'camera_id and chunk_number are required'}), 400
        
        try:
            camera_id = int(camera_id)
            chunk_number = int(chunk_number)
        except ValueError:
            return jsonify({'error': 'camera_id and chunk_number must be integers'}), 400
        
        # Verificar que la cámara esté en rango
        if camera_id >= current_session['cameras_count']:
            return jsonify({'error': f'Invalid camera_id. Max: {current_session["cameras_count"]-1}'}), 400
        
        # Guardar archivos
        patient_id = current_session['patient_id']
        session_id = current_session['session_id']
        
        # Comprobación de que las cámaras están grabando bien (cuando llega un chunk 2, que tenemos al menos el chunk 0 de todas)
        global chunk_2_verified
        if chunk_number == 2 and not chunk_2_verified:
            chunk_2_verified = True
            logger.info("Verificando integridad de chunks de cámaras al recibir chunk 2...")
            
            integrity_ok = _check_camera_chunks_integrity(patient_id, session_id, current_session['cameras_count'])
            if not integrity_ok:
                logger.error("FALLO DE CÁMARAS DETECTADO - Algunas cámaras no enviaron chunks correctamente")
                _cancel_session_due_to_camera_failure()
                return jsonify({
                    'error': 'CAMERA_FAILURE_DETECTED',
                    'message': 'Fallo crítico de cámaras detectado. Sesión cancelada automáticamente.',
                    'action_required': 'Desconectar y conectar el switch de las cámaras antes de intentar una nueva grabación.'
                }), 500
        
        save_dir = data_config.unprocessed_dir / f"patient{patient_id}" / f"session{session_id}" / f"camera{camera_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar archivo de color
        color_filename = f"{chunk_number}_color.mp4"
        color_path = save_dir / color_filename
        file_color.save(str(color_path))
        
        # Guardar archivo de profundidad
        depth_filename = f"{chunk_number}_depth.npy"
        depth_path = save_dir / depth_filename
        file_depth.save(str(depth_path))
        
        color_size = color_path.stat().st_size
        depth_size = depth_path.stat().st_size
        
        logger.info(f"Chunk recibido - Cámara: {camera_id}, Chunk: {chunk_number}")
        logger.info(f"  Color: {color_size} bytes ({color_filename})")
        logger.info(f"  Depth: {depth_size} bytes ({depth_filename})")
        
        

        # Inicializar solo una vez el coordinador de los detectores 2D (si no se usa el lock, se inicializa varias veces y falla)
        with coordinator_lock:
            if not pose_coordinator.initialized:
                logger.info("Inicializando coordinador de procesamiento de pose...")
                initialization_success = pose_coordinator.initialize_all()
                
                # Permitir continuar aunque algunos detectores fallen, siempre que al menos uno funcione
                if not initialization_success:
                    logger.error("Error inicializando coordinador de pose - ningún detector se inicializó correctamente")
                    return jsonify({'error': 'Error initializing pose processing coordinator - no detectors available'}), 500
                else:
                    logger.info("Coordinador inicializado correctamente con al menos un detector")
        
        # Procesar todos los chunks de todas las cámaras
        processing_results = None
        logger.info(f"Procesando chunk {chunk_number} de cámara {camera_id}")
        
        # Usar semáforo para permitir chunks procesándose simultáneamente según configuración (1 o 2 GPUs)
        with processing_semaphore:
            logger.info(f"Iniciando procesamiento paralelo de chunk {chunk_number} cámara {camera_id} (máximo {gpu_config.max_concurrent_chunks} simultáneos)")
            
            # Procesar este chunk con todos los detectores tradicionales
            chunk_id = str(chunk_number)
            
            processing_results = pose_coordinator.process_chunk(
                video_path=color_path,
                patient_id=patient_id,
                session_id=session_id,
                camera_id=camera_id,
                chunk_id=chunk_id
            )
            
            
            success_count = sum(processing_results.values())
            logger.info(f"Chunk {chunk_number} cámara {camera_id} procesado - {success_count}/{len(processing_results)} detectores exitosos")
            

            # Procesamiento avanzado con visualización si TRT detector está disponible
            advanced_results = None
            try:
                # Cargar frames de profundidad
                depth_frames = np.load(str(depth_path))
                if not isinstance(depth_frames, list):
                    depth_frames = [depth_frames] if depth_frames.ndim == 2 else list(depth_frames)
                
                # Directorio para guardar resultados avanzados
                advanced_output_dir = data_config.processed_dir / f"patient{patient_id}" / f"session{session_id}" / "advanced_analysis"
                advanced_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Obtener TRT detector del coordinador
                trt_detector = next((detector for detector in pose_coordinator.detectors if isinstance(detector, TrtDetector)), None)
                if trt_detector and trt_detector.is_initialized:
                    logger.info(f"Iniciando procesamiento avanzado para chunk {chunk_number} cámara {camera_id}")
                    
                    with gait_tracker_lock:
                        # Procesar con visualización avanzada
                        advanced_results = process_chunk_with_advanced_visualization(
                            trt_detector=trt_detector,
                            manual_action_detector=posture_classifier,
                            gait_tracker=gait_tracker,
                            video_path=color_path,
                            depth_frames=depth_frames,
                            camera_id=camera_id,
                            chunk_id=chunk_id,
                            output_dir=advanced_output_dir
                        )
                        
                        logger.info(f"Procesamiento avanzado completado para chunk {chunk_number} cámara {camera_id}")
                        if advanced_results.get('success'):
                            logger.info(f"Video anotado guardado: {advanced_results.get('video_path')}")
                            logger.info(f"Distancia total de marcha: {advanced_results.get('total_gait_distance', 0):.3f}m")
                else:
                    logger.warning("TRT detector no disponible para procesamiento avanzado")
                    
            except Exception as e:
                logger.error(f"Error en procesamiento avanzado: {e}")
                advanced_results = {'success': False, 'error': str(e)}
            
            logger.info(f"Procesamiento paralelo completado para chunk {chunk_number} cámara {camera_id}")
            
            # Registrar finalización del chunk en ensemble processor. Cuando se haya procesado el último chunk de todas las cámaras, se iniciará automáticamente el ensemble.
            # COMENTADO -------------------------------------------------------------
            #chunk_completed = ensemble_processor.register_chunk_completion(
            #    patient_id, session_id, f"camera{camera_id}", chunk_number
            #)
            #if chunk_completed:
            #    logger.info(f"¡Chunk final completado por todas las cámaras! Ensemble iniciado automáticamente")
            # -----------------------------------------------------------------------

        
        response_data = {
            'status': 'chunk_received',
            'camera_id': camera_id,
            'chunk_number': chunk_number,
            'chunk_id': chunk_id,
            'color_file_path': str(color_path),
            'depth_file_path': str(depth_path),
            'color_file_size': color_size,
            'depth_file_size': depth_size,
            'message': 'Chunk with color and depth data saved successfully'
        }
        
        # Agregar información de procesamiento.
        response_data['processing_results'] = processing_results
        response_data['processed'] = True
        response_data['successful_detectors'] = sum(processing_results.values())
        response_data['total_detectors'] = len(processing_results)
        
        # Agregar resultados del procesamiento avanzado
        if advanced_results:
            response_data['advanced_processing'] = advanced_results
            if advanced_results.get('success'):
                response_data['annotated_video_path'] = advanced_results.get('video_path')
                response_data['gait_analysis'] = {
                    'chunk_distance': advanced_results.get('gait_distance', 0),
                    'total_distance': advanced_results.get('total_gait_distance', 0),
                    'trajectory_points': advanced_results.get('gait_trajectory_points', 0)
                }
                response_data['action_detection'] = {
                    'processed_frames': advanced_results.get('processed_frames', 0),
                    'results_count': len(advanced_results.get('action_results', []))
                }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error recibiendo chunk: {str(e)}")
        return jsonify({'error': f'Failed to receive chunk: {str(e)}'}), 500

@app.route('/api/gait/stats', methods=['GET'])
def get_gait_stats():
    """
    Obtener estadísticas actuales del gait tracker
    """
    try:
        with gait_tracker_lock:
            stats = gait_tracker.stats()
        
        return jsonify({
            'status': 'success',
            'gait_stats': stats,
            'session_info': {
                'patient_id': current_session.get('patient_id'),
                'session_id': current_session.get('session_id'),
                'is_active': current_session.get('is_active', False)
            }
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo estadísticas de gait: {str(e)}")
        return jsonify({'error': f'Failed to get gait stats: {str(e)}'}), 500

@app.route('/api/gpu/status', methods=['GET'])
def get_gpu_status():
    """
    Obtener estado actual de las GPUs
    """
    try:
        if not pose_coordinator.initialized:
            return jsonify({
                'coordinator_initialized': False,
                'message': 'Pose coordinator not initialized'
            })
        
        gpu_status = pose_coordinator.get_gpu_status()
        
        return jsonify({
            'coordinator_initialized': True,
            'gpu_status': gpu_status,
            'processing_mode': gpu_status.get('mode', 'unknown')
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo estado de GPU: {str(e)}")
        return jsonify({'error': f'Failed to get GPU status: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    """Manejar archivos demasiado grandes"""
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(e):
    """Manejar rutas no encontradas"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Manejar errores internos"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info(f"Iniciando servidor...")
    logger.info(f"Puerto: {server_config.port}")
    logger.info(f"Directorio de datos: {data_config.base_data_dir}")
    logger.info(f"GPUs configuradas: {gpu_config.available_gpus}")
    logger.info(f"Máximo chunks concurrentes: {gpu_config.max_concurrent_chunks}")
    
    app.run(
        host=server_config.host,
        port=server_config.port, 
        debug=server_config.debug
    )
