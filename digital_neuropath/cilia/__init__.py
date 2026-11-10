"""3D detection and measurement of primary cilia and centrosomes in human postmortem cortex.
load -> segment_cilia -> segment_mtocs -> assign_to_neurons -> summarize_cilia.
Each pipeline is one file read top to bottom, so the method a number came from is on the page.
"""
from digital_neuropath.cilia.cilia import segment_cilia
from digital_neuropath.cilia.link import assign_to_neurons
from digital_neuropath.cilia.load import Fov, load, read_table
from digital_neuropath.cilia.measure import summarize_cilia
from digital_neuropath.cilia.mtocs import segment_mtocs

__version__ = "2.0.0.dev0"

__all__ = ["Fov", "load", "segment_cilia", "segment_mtocs", "assign_to_neurons",
           "summarize_cilia", "read_table"]
