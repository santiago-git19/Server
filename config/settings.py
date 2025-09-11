from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path

# Directorio base del proyecto
BASE_DIR = Path(__file__).parent.parent

@dataclass
class ServerConfig:
    """Configuración del servidor Flask"""
    host: str = '0.0.0.0'
    port: int = 5000
    debug: bool = True
    max_content_length: int = 100 * 1024 * 1024  # 100MB
    upload_folder: str = str(BASE_DIR / "data" / "unprocessed")


@dataclass
class GPUConfig:
    """Configuración de GPUs disponibles para procesamiento"""
    # Lista de GPUs disponibles (IDs que se pueden usar)
    # Ejemplos:
    # [0] - Solo GPU 0 disponible
    # [1] - Solo GPU 1 disponible  
    # [0, 1] - Ambas GPUs disponibles (por defecto)
    # [] - Sin GPUs (usar CPU)
    available_gpus: list = field(default_factory=lambda: [0, 1])
    
    # Número máximo de chunks procesándose simultáneamente
    # Se ajusta automáticamente al número de GPUs disponibles
    max_concurrent_chunks: int = None
    
    def __post_init__(self):
        """Ajustar configuración después de inicialización"""
        # Si no se especifica max_concurrent_chunks, usar el número de GPUs disponibles
        if self.max_concurrent_chunks is None:
            self.max_concurrent_chunks = max(1, len(self.available_gpus))
    
    def get_gpu_usage_dict(self) -> Dict[int, bool]:
        """Obtener diccionario de uso de GPUs inicializado"""
        return {gpu_id: False for gpu_id in self.available_gpus}
    
    def is_gpu_available(self, gpu_id: int) -> bool:
        """Verificar si una GPU específica está disponible"""
        return gpu_id in self.available_gpus


@dataclass
class DataConfig:
    """Configuración de directorios de datos"""
    base_data_dir: Path = BASE_DIR / "data"
    unprocessed_dir: Path = base_data_dir / "unprocessed"
    processed_dir: Path = base_data_dir / "processed"
    
    # Directorios específicos de datos procesados
    keypoints_2d_dir: Path = processed_dir / "2D_keypoints"
    keypoints_3d_dir: Path = processed_dir / "3D_keypoints"
    photos_dir: Path = processed_dir / "photos_from_video"
    
    # Directorio de logs
    logs_dir: Path = BASE_DIR / "logs"
    
    # Directorio para videos anotados
    annotated_videos_dir: Path = processed_dir / "annotated_videos"
    
    def ensure_directories(self):
        """Crear todos los directorios necesarios"""
        dirs_to_create = [
            self.base_data_dir,
            self.unprocessed_dir,
            self.processed_dir,
            self.keypoints_2d_dir,
            self.keypoints_3d_dir,
            self.photos_dir,
            self.logs_dir,
            self.annotated_videos_dir
        ]
        
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)

@dataclass
class MMPoseConfig:
    """Configuración para modelos MMPose"""
    models_dir: Path = BASE_DIR / "mmpose_models"
    configs_dir: Path = models_dir / "configs"
    checkpoints_dir: Path = models_dir / "checkpoints"
    
    # Configuraciones específicas de cada detector
    vitpose: Dict[str, str] = field(default_factory=lambda: {
        'pose2d': 'configs/pose2d/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py',
        'pose2d_weights': 'checkpoints/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth',
        'device': 'cuda:0'
    })
    
    mspn: Dict[str, str] = field(default_factory=lambda: {
        'pose2d': 'configs/pose2d/td-hm_4xmspn50_8xb32-210e_coco-256x192.py',
        'pose2d_weights': 'checkpoints/4xmspn50_coco_256x192-7b837afb_20201123.pth',
        'device': 'cuda:0'
    })
    
    hrnet: Dict[str, str] = field(default_factory=lambda: {
        'pose2d': 'configs/pose2d/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py',
        'pose2d_weights': 'checkpoints/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth',
        'device': 'cuda:0'
    })
    
    csp: Dict[str, str] = field(default_factory=lambda: {
        'pose2d': 'configs/pose2d/cspnext-m_udp_8xb64-210e_coco-wholebody-256x192.py',
        'pose2d_weights': 'checkpoints/cspnext-m_udp-coco-wholebody_pt-in1k_210e-256x192-320fa258_20230123.pth',
        'device': 'cuda:0'
    })
    
    def ensure_directories(self):
        """Crear directorios para modelos"""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class CameraIntrinsicsConfig:
    """Configuración de parámetros intrínsecos para cada cámara"""
    
    # Parámetros intrínsecos por cámara (Orbbec Gemini 335Le)
    # Formato: camera_id -> {'fx': float, 'fy': float, 'cx': float, 'cy': float}
    intrinsics: Dict[int, Dict[str, float]] = field(default_factory=lambda: {
        0: {  # Cámara 0
            'fx': 375.0805358886719,
            'fy': 375.0805358886719,
            'cx': 320.6000061035156,
            'cy': 241.5
        },
        1: {  # Cámara 1
            'fx': 374.8923645019531,
            'fy': 374.8923645019531,
            'cx': 319.2150573730469,
            'cy': 240.8741760253906
        },
        2: {  # Cámara 2
            'fx': 375.2341766357422,
            'fy': 375.2341766357422,
            'cx': 321.1058044433594,
            'cy': 242.0329895019531
        },
        3: {  # Cámara 3
            'fx': 374.7654418945312,
            'fy': 374.7654418945312,
            'cx': 318.9876098632812,
            'cy': 241.2156982421875
        },
        4: {  # Cámara 4
            'fx': 375.4127502441406,
            'fy': 375.4127502441406,
            'cx': 322.0945434570312,
            'cy': 242.7083740234375
        }
    })
    
    def get_intrinsics(self, camera_id: int) -> Dict[str, float]:
        """
        Obtener parámetros intrínsecos para una cámara específica.
        
        Args:
            camera_id: ID de la cámara (0-4)
            
        Returns:
            Diccionario con parámetros intrínsecos {'fx', 'fy', 'cx', 'cy'}
            
        Raises:
            ValueError: Si camera_id no está configurado
        """
        if camera_id not in self.intrinsics:
            raise ValueError(f"Camera ID {camera_id} not configured. Available cameras: {list(self.intrinsics.keys())}")
        
        return self.intrinsics[camera_id].copy()
    
    def get_available_cameras(self) -> list:
        """Obtener lista de cámaras configuradas"""
        return list(self.intrinsics.keys())
    
    def set_intrinsics(self, camera_id: int, fx: float, fy: float, cx: float, cy: float):
        """
        Configurar parámetros intrínsecos para una cámara específica.
        
        Args:
            camera_id: ID de la cámara
            fx, fy: Distancia focal en píxeles
            cx, cy: Centro óptico en píxeles
        """
        self.intrinsics[camera_id] = {
            'fx': float(fx),
            'fy': float(fy),
            'cx': float(cx),
            'cy': float(cy)
        }

@dataclass
class ProcessingConfig:
    """Configuración para procesamiento de video"""
    # Configuración de videos anotados
    save_annotated_videos: bool = False  # Si guardar videos con keypoints dibujados

@dataclass
class EnsembleConfig:
    """Configuración para procesamiento de ensemble"""    
    # Número mínimo de detectores requeridos para generar ensemble
    min_detectors_required: int = 1

# Instancias globales de configuración
server_config = ServerConfig()
gpu_config = GPUConfig()
camera_intrinsics_config = CameraIntrinsicsConfig()
processing_config = ProcessingConfig()
ensemble_config = EnsembleConfig()
data_config = DataConfig()
mmpose_config = MMPoseConfig()

# Inicializar directorios al importar
data_config.ensure_directories()
mmpose_config.ensure_directories()
