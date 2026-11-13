"""yolo-bench — training, export e benchmark sistematico di YOLO26 su edge."""

__version__ = "0.1.0"

from omegaconf import OmegaConf

# ${meno:a,b,c} = a - b - c, arrotondato a 0,1 ms: budget del modello dal profilo (conf/target/)
if not OmegaConf.has_resolver("meno"):
    OmegaConf.register_new_resolver("meno", lambda a, *b: round(float(a) - sum(map(float, b)), 1))
# ${ms_per_fps:30} = 33.3: il tempo di un fotogramma
if not OmegaConf.has_resolver("ms_per_fps"):
    OmegaConf.register_new_resolver("ms_per_fps", lambda fps: round(1000 / float(fps), 1))
