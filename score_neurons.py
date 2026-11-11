# digital_neuropath: per-neuron scoring of 3D confocal stacks

# =============================== INPUT ===============================

# Your files, where they already are, one per line. When they share a folder, write it once
# here and each file as f"{FOLDER}/<file name>"; otherwise give each file's full path.
# Masks pair with images by position (the 1st mask belongs to the 1st image), so each mask
# list has as many files as IMAGES.
FOLDER = "/path/to/your/files"

IMAGES = [
    f"{FOLDER}/image1.tif",
]   # .tif (OME or ImageJ) or .lif

NEURON_MASKS = [
    f"{FOLDER}/image1_neuron_mask.tif",
]   # labelled neuron masks (.tif or .cmap); [] = click each neuron

NUCLEI_MASKS = [
    f"{FOLDER}/image1_nuclei.tif",
]   # labelled nucleus masks (.tif), optional; [] = none

OUTPUT = "output"   # folder where the tool writes everything; tables in output/results/

# Channel of each stain in your images; the first channel is 0
DAPI = 0
NEUN = 1

# Stains, scored one after the other in this order: "name": (channel, [categories])
# "unsure" and "not scoreable" are added to every list; [] = loaded, not scored.
# A neuron called "not scoreable" on one stain is not shown on the next.
STAINS = {
    "tdp43": (2, ["Normal", "Nuclear clearance", "NCI"]),
    "ac3": (3, ["yes, normal", "abnormally long", "abnormally short", "not found"]),
}

# Other stains to load in a stain's window, e.g. {"tdp43": ["acvr1c"]}. Seeing another stain
# can bias the call; {} keeps each window to DAPI, NeuN and its own stain.
ALSO_SHOW = {}

# True : every session, neurons of all images in a new random order, no image names
# False: image by image, in order, image name and neuron shown
BLIND = True
SCORE = list(STAINS)   # stains to score this session; the others keep their calls and are not shown

# ======================================================================

RUN = ["neurons", "scoring"]   # add "mtocs", "cilia", "link" for 3D cilia (needs "ac3" and "pcnt")
PARAMS = {}

if __name__ == "__main__":
    from digital_neuropath import app
    app.run_from_settings(__file__)
