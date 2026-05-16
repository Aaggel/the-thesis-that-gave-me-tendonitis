# οι βιβλιοθήκες μας / our libraries ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import os # για πρόσβαση στο λειτουργικό σύστημα
import hashlib # για κρυπτογράφηση του cache
import numpy as np # για τα μαθηματικά
import pandas as pd # για να αποθηκεύουμε σε dataframe
import pydicom # για να μπορεί να διαβάζει αρχεία dicom
import multiprocessing # για να μπορεί να κάνει πολλά πράγματα ταυτόχρονα
import torch # για το AI
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision.models as models 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm # για την μπάρα στο output / for the progress bar
from sklearn.metrics import confusion_matrix, classification_report # για τις μετρήσεις
import matplotlib.pyplot as plt # για τα γραφήματα
from idc_index import IDCClient # για την σύνδεση με την βάση του idc
import wandb # για να αποθηκεύει τα αποτελέσματα
import shutil # για να χειρίζεται φακέλους και αρχεία
import psutil # για να βλέπει τα κομμάτια του υπολογιστή

# μερικές ρυθμίσεις / some settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__": # για να μην κάνει λούπες στα windows
    multiprocessing.freeze_support()

# default τιμές
DEFAULT_MAX_SLICES = 4 # κομμάτια από βάση
DEFAULT_BATCH_SIZE = 4 # κομμάτια για εκπαίδευση
DEFAULT_NUM_WORKERS = 4 # υποδιεργασίες
DEFAULT_LIMIT = 30 # όριο για να μην κατεβάζει ολόκληρες τις συλλογές

# για να μπορεί να δει το μηχάνημα
def get_hardware_info():
     # για να δει την CPU
    cpu_cores = multiprocessing.cpu_count()
    # για να δει την RAM (σε gigabyte)
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    # για να δει τον ελευθερο χώρο στον δίσκο (σε gigabyte)
    disk_gb = shutil.disk_usage(".").free / (1024 ** 3)
    # για να δει την GPU VRAM (σε gigabyte)
    gpu_gb = 0
    if torch.cuda.is_available(): # η gpu μόνο αν το μηχάνημα έχει cuda
        gpu_gb = (torch.cuda.get_device_properties(0).total_memory/ (1024 ** 3))
    return {"cpu": cpu_cores, "ram": ram_gb, "disk": disk_gb, "gpu": gpu_gb}

# για να μπορεί να προσαρμοστεί στο μηχάνημα
def auto_settings():
    hw = get_hardware_info()
    # αρχικά με τα default
    settings = {"MAX_SLICES": DEFAULT_MAX_SLICES, "BATCH_SIZE": DEFAULT_BATCH_SIZE, "NUM_WORKERS": DEFAULT_NUM_WORKERS, "LIMIT": DEFAULT_LIMIT}
    # για πιο αδύναμα συστήματα
    if hw["ram"] < 8 or hw["disk"] < 20:
        settings["LIMIT"] = 10
        settings["BATCH_SIZE"] = 2
        settings["MAX_SLICES"] = 2
        settings["NUM_WORKERS"] = 1
    # για πιο δυνατά συστήματα
    elif hw["ram"] >= 16 and hw["cpu"] >= 8:
        settings["LIMIT"] = 60
        settings["NUM_WORKERS"] = 6
    # για συστήματα με καλή gpu
    if hw["gpu"] >= 8:
        settings["BATCH_SIZE"] = 8
        settings["MAX_SLICES"] = 6
        settings["LIMIT"] = 100
    # για συστήματα με πάρα πολύ καλή gpu
    if hw["gpu"] >= 12:
        settings["BATCH_SIZE"] = 12
        settings["MAX_SLICES"] = 8
        settings["LIMIT"] = 150
    # για επαγγελματικά συστήματα
    if hw["gpu"] >= 24 and hw["ram"] >= 64:
        settings["BATCH_SIZE"] = 16
        settings["MAX_SLICES"] = 12
        settings["LIMIT"] = 250
        settings["NUM_WORKERS"] = 12
    # για έξτρα ασφαλεια
    settings["NUM_WORKERS"] = min(settings["NUM_WORKERS"], max(1, hw["cpu"] - 1))
    return settings

# εφαρμόζουμε τις ρυθμίσεις
SETTINGS = auto_settings()
MAX_SLICES = SETTINGS["MAX_SLICES"]
BATCH_SIZE = SETTINGS["BATCH_SIZE"]
NUM_WORKERS = SETTINGS["NUM_WORKERS"]
LIMIT = SETTINGS["LIMIT"]
# τις τυπώνουμε
HW = get_hardware_info()
print("\n﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏")
print("AUTO HARDWARE CONFIG")
print("﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏")
print(f"CPU CORES : {HW['cpu']}")
print(f"RAM (GB)  : {HW['ram']:.1f}")
print(f"DISK (GB) : {HW['disk']:.1f}")
print(f"GPU (GB)  : {HW['gpu']:.1f}")
print("\nACTIVE SETTINGS")
print(f"MAX_SLICES : {MAX_SLICES}")
print(f"BATCH_SIZE : {BATCH_SIZE}")
print(f"NUM_WORKERS: {NUM_WORKERS}")
print(f"LIMIT      : {LIMIT}")
print("﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏﹏\n")

# το "σκονάκι" για τα training data
COLLECTION_LABELS = {
    "4d_lung": {"organ": "lung", "tumor": "tumor"},
    "acrin_flt_breast": {"organ": "breast", "tumor": "tumor"},
    "breast_diagnosis": {"organ": "breast", "tumor": "tumor"},
    "cmb_ov": {"organ": "ovary", "tumor": "tumor"},
    "colorectal_liver_metastases": {"organ": "skin", "tumor": "tumor"},
    "cptac_cm": {"organ": "skin", "tumor": "tumor"},
    "prostate_mri": {"organ": "prostate", "tumor": "tumor"},
    "pancreas_ct": {"organ": "pancreas", "tumor": "no_tumor"},
    "prostatex": {"organ": "prostate", "tumor": "no_tumor"},
    "spine_mets_ct_seg": {"organ": "bone", "tumor": "tumor"},
    "covid_19_ar": {"organ": "lung", "tumor": "no_tumor"},
    "remind": {"organ": "brain", "tumor": "tumor"}
    }

# για να βλέπει αν υπάρχει cuda
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# για την προσωρινή μνήμη (cache)
CACHE_DIR = "dicom_cache" # για να μην ξανακατεβάζει ολόκληρα τα αρχεία
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_key(path): # κρυπτογράφηση με κλειδί για πιο πολύ ασφάλεια
    return hashlib.md5(path.encode()).hexdigest()

def load_cached_dicom(path): # για να βρίσκει που είναι αποθηκευμένο το cache
    key = cache_key(path)
    cache_path = os.path.join(CACHE_DIR, key + ".pt")
    if os.path.exists(cache_path): # αν υπάρχει ήδη η διαδρομή
        try:
            return torch.load(cache_path) # προσπαθεί να το φορτώσει
        except:
            pass # αν δεν μπορεί, προχωράει
    try:
        dcm = pydicom.dcmread(path) # προσπαθεί να διαβάσει τα αρχεία
        if "PixelData" not in dcm: # αν δεν υπάρχουν μέσα πίξελ
            raise ValueError() # δεν το δέχεται, γιατί πάει να πει δεν είναι εικόνα
        img = dcm.pixel_array.astype(np.float32) # μετατροπή των πίξελ σε αριθμό
        if len(img.shape) != 2: # αμα η εικόνα δεν έχει δύο διαστάσεις
            raise ValueError() # πάλι δεν την δέχεται
        img -= img.min() # κάνει την minimum τιμή μηδέν
        img /= (img.max() + 1e-6) # για να είναι όλα τα πίξελ από 0.0 μέχρι 1.0
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0) # κάνει τους αριθμούς tensor και βάζει δύο ψεύτικες διαστάσεις
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False) # για να είναι όλες οι εικόνες στο ίδιο μέγεθος
        img = img.squeeze(0).repeat(3, 1, 1).float() # αφαιρεί την μία ψεύτικη διάσταση και αντιγράφει το κανάλι 3 φορές για να μοιάζει με έγχρωμη rgb εικόνα (το θέλουμε για το resnet)
    except:
        img = torch.zeros((3, 224, 224)) # αλλιώς φτιάχνει μία μαύρη εικόνα για να μην κρασάρει
    try:
        torch.save(img, cache_path) # προσπαθεί να αποθηκεύσει τις επεξεργασμένες εικόνες
    except:
        pass # αν δεν μπορεί, συνεχίζει
    return img # μας το φέρνει πίσω

# Για να μπορεί να διαβάζει τις ταμπέλες ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def infer_organ(row):
    body = str(row.get("body_part", "")).upper() # κοιτάει το πεδίο για τα όργανα και κάνει τα γράμματα κεφαλαία

    # λεξικό με συνώνυμα γιατί στο idc τα γράφουν όπως να ναι και δεν τα καταλαβαίνει
    direct_map = {
        "BRAIN": "brain", "HEAD": "brain", "SKULL": "brain",
        "CHEST": "lung", "LUNG": "lung", "THORAX": "lung",
        "BREAST": "breast",
        "ABDOMEN": "liver", "LIVER": "liver", "HEPATIC": "liver",
        "PROSTATE": "prostate",
        "PANCREAS": "pancreas",
        "COLON": "colon", "RECTUM": "colon", "RECTAL": "colon", "BOWEL": "colon",
        "OVARY": "ovary", "OVARIAN": "ovary",
        "KIDNEY": "kidney", "RENAL": "kidney",
        "SPINE": "bone", "VERTEBRA": "bone", "SKELETAL": "bone", "BONE": "bone", "MARROW": "bone",
        "BLOOD": "blood", "LYMPHOMA": "blood", "LEUKEMIA": "blood",
        "SKIN": "skin", "CUTANEOUS": "skin", "MELANOMA": "skin"
    }
    for k, v in direct_map.items(): # για να μπορεί να συνδέσει τις λέξεις
        if k in body:
            return v

    # για καλό και για κακό κοιτάμε και άλλα πεδία, μήπως τα έχουν γράψει εκεί
    text = (str(row.get("study", "")) + " " + str(row.get("series_desc", ""))).lower() # εδώ κάνουμε τα γράμματα μικρά
    # κι εδώ συνώνυμα
    mapping = { 
        "brain": ["brain", "head", "neuro", "cranial"],
        "lung": ["lung", "chest", "thorax", "pulmonary"],
        "liver": ["liver", "hepatic"],
        "breast": ["breast", "mammography"],
        "prostate": ["prostate"],
        "pancreas": ["pancreas", "pancreatic"],
        "colon": ["colon", "colorectal", "bowel", "rectal"],
        "ovary": ["ovary", "ovarian"],
        "kidney": ["kidney", "renal"],
        "bone": ["bone", "skeletal", "spine", "vertebra", "marrow"],
        "blood": ["blood", "lymphoma", "leukemia"],
        "skin": ["skin", "melanoma", "cutaneous"]
    }
    # κοιτάμε και εδώ το λεξικό
    for label, keys in mapping.items():
        if any(k in text for k in keys):
            return label
    return "unknown" # και άμα δεν είναι τίποτα, λέει ότι δεν ξέρει

# τα ίδια κάνουμε και για τους όγκους
def infer_tumor(row):
    text = (str(row.get("study", "")) + " " + str(row.get("series_desc", "")) + " " + str(row.get("collection", ""))).lower()
    # προσπάθησα με λεξικά πάλι
    tumor_words = ["tumor", "mass", "lesion", "cancer", "malignant", "neoplasm", "metastasis",
                   "melanoma", "sarcoma", "carcinoma", "glioma", "adenoma", "lymphoma", "leukemia"]
    cancer_collections = ["cptac", "tcga", "cancer", "metastasis", "oncology", "sarcoma", "melanoma", "lymphoma"]
    if any(w in text for w in tumor_words):
        return "tumor"
    if any(w in text for w in cancer_collections):
        return "tumor"
    # αλλά επειδή δεν δούλευε πολύ καλά, είπα να το πάω ανάποδα
    if "non-cancer" in text:
        return "no_tumor"
    return "tumor"
    # οπότε τώρα υποθέτει ότι όλα είναι καρκίνοι, εκτός άμα το γράφει ξεκάθαρα ότι δεν είναι

# Κατεβάζουμε τα δεδομένα από το IDC ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def download_collection(name, limit=LIMIT): # δεν ξεχνάμε το όριο
    client = IDCClient() # σύνδεση εδώ
    df = client.index # βάζουμε το ευερετήριο σε dataframe
    subset = (df[df["collection_id"] == name].drop_duplicates("SeriesInstanceUID").head(limit)) # μόνο με συγκεκριμένα id κ βγάζουμε και τα διπλότυπα
    out = os.path.join("idc_data", name) # τα βάζουμε όλα μαζί
    os.makedirs(out, exist_ok=True) # τα βάζουμε σε έναν φάκελο
    if any(f.endswith(".dcm") for _, _, files in os.walk(out) for f in files): # αν υπάρχουν ήδη αρχεία dicom στον φάκελο
        print(f"[SKIP DOWNLOAD] {name}") # το παραλείπει γιατί σημαίνει ότι λογικά τα έχει κατεβάσει ήδη σε προηγούμενο run
        return out 
    try:
        client.download_dicom_series(subset["SeriesInstanceUID"].tolist(), out) # αν όλα πάνε καλά, τα κάνει λίστα και αρχίζει να τα κατεβάζει
    except Exception as e:
        print(f"[SKIP DATASET] {name} -> {e}") # αν όχι, μας λέει τι πήγε στραβά
        return None
    return out

# κατασκευάζουμε το dataset / we build the dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def build_dataset(root):
    rows = [] # οι γραμμές αρχικά κενές
    for r, _, files in os.walk(root): # κοιτάει όλους τους φακέλους και τους υποφακέλους
        for f in files:
            if not f.endswith(".dcm"): # αν κάτι δεν είναι αρχείο dicom το παραλείπουμε
                continue
            path = os.path.join(r, f) # κρατάμε την διαδρομή
            try:
                dcm = pydicom.dcmread(path) # διαβάζουμε τα μεταδεδομένα
                if "PixelData" not in dcm: # αν δεν έχει πίξελ το παραλείπουμε γιατί πάει να πει δεν είναι εικόνα
                    continue
                if int(getattr(dcm, "NumberOfFrames", 1)) > 1: # αν έχει πάνω από ένα καρέ παραλείπουμε γιατί σημαίνει είναι βίντεο
                    continue
                if int(getattr(dcm, "SamplesPerPixel", 1)) > 1: # παραλείπουμε και έγχρωμες εικόνες γιατί οι ακτινογραφίες είναι ασπρόμαυρες
                    continue
                rows.append({ # συμπληρώνουμε τις πληροφορίες
                    "filepath": path,
                    "series_id": getattr(dcm, "SeriesInstanceUID", "unknown"),
                    "study": getattr(dcm, "StudyDescription", ""),
                    "series_desc": getattr(dcm, "SeriesDescription", ""),
                    "body_part": getattr(dcm, "BodyPartExamined", ""),
                    "collection": os.path.basename(root)
                })
            except: # αν κάτι πάει στραβά συνεχίζουμε
                continue
    df = pd.DataFrame(rows) # τα αποθηκεύουμε στο dataframe
    print("\n[DATASET] valid:", len(df)) # τυπώνουμε πόσα αρχεία κρατήσαμε
    if len(df) == 0:
        return df # αν δεν κρατήσαμε τίποτα, επιστρέφει ένα άδειο dataframe
    collection = os.path.basename(root) 
    if collection in COLLECTION_LABELS: # κοιτάει αν υπάρχει το όργανο ή η κατάσταση του στις ταμπελίτσες
        df["organ"] = COLLECTION_LABELS[collection]["organ"]
        df["tumor"] = COLLECTION_LABELS[collection]["tumor"]
    else: # αν δεν υπάρχει προσπαθεί να το βρει με τα λεξικά (πιο πάνω)
        df["organ"] = df.apply(infer_organ, axis=1)
        df["tumor"] = df.apply(infer_tumor, axis=1)
    return df

def build_multi(names):
    dfs = [] # εδώ θα τα βάλουμε όλα μαζί σε ένα
    for n in names: # για όλες τις συλλογές στην λίστα
        print("\n𓂃⋆.˚", n, "˚.⋆𓂃")  # εδώ τυπώνουμε για να τα βλέπουμε και στην κονσόλα
        folder = download_collection(n) # καλούμε την συνάρτηση
        if folder is None: # αν δεν υπάρχει κάποιος φάκελος
            continue # παραλείπουμε
        df = build_dataset(folder) # καλούμε και την άλλη συνάρτηση (από πάνω)
        if len(df) < 10: # αν δεν έχουμε έστω 10 έγκυρα αρχεία σε μία συλλογή
            continue # την παραλείπουμε
        dfs.append(df) # αλλιώς την προσθέτουμε στο συνολικό
    return pd.concat(dfs).reset_index(drop=True) if dfs else pd.DataFrame() # αν έχουμε έστω και μία έγκυρη συλλογή κάνουμε συνένωση

#  dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class VolumeDataset(Dataset):
    def __init__(self, df):
        self.df = df 
        self.groups = df.groupby("series_id") # ομαδοποιούμε ανα σειρά
        self.keys = list(self.groups.groups.keys()) # κρατάμε λίστα με τα id των σειρών
        # λεξικά επειδή τα μοντέλα της Pytorch καταλαβαίνουν μ΄νο αριθμούς
        self.organ_map = { 
            "brain": 0, "lung": 1, "liver": 2, "breast": 3, "prostate": 4, "pancreas": 5, "colon": 6, 
            "ovary": 7, "kidney": 8, "bone": 9, "blood": 10, "skin": 11, "unknown": 12
        }
        self.tumor_map = {"no_tumor": 0, "tumor": 1}
    def __len__(self):
        return len(self.keys) # μας λέει πόσοι τόμοι υπάρχουν, δηλ. ολοκληρωμένα σείγματα
    def __getitem__(self, i):
        g = self.groups.get_group(self.keys[i]).sort_values("filepath") # για να μπουν με τη σειρά
        imgs = [load_cached_dicom(r["filepath"]) for _, r in g.head(MAX_SLICES).iterrows()] # φορτώνει cache για πιο γρήγορα
        while len(imgs) < MAX_SLICES: # για να έχουν όλοι οι τόμοι το ίδιο μέγεθος, αλλιώς δεν δουλεύει η pytorch
            imgs.append(imgs[-1].clone())
        x = torch.stack(imgs) # ενώνουμε τις 2D εικόνες για να φτιάξουμε 3D tensor!
        r0 = g.iloc[0] # Βλέπει τις πληροφορίες του τόμου από την πρώτη γραμμή
        return (x, torch.tensor(self.organ_map.get(r0["organ"], 12)), torch.tensor(self.tumor_map.get(r0["tumor"], 1)))
        # επιστρέφει tensor, όργανο κ κατάσταση

# Φτιάχνουμε το μοντέλο μας! :) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Model(nn.Module):
    def __init__(self):
        super().__init__() # για να το βλέπει από όλες τις συναρτήσεις
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # φορτώνουμε το μοντέλο resnet18 (είναι ήδη προεκπαιδευμένο στο imagenet)
        self.backbone = nn.Sequential(*list(base.children())[:-1]) # αφαιρούμε classification layer του resnet18 κ κρατάμε μόνο feature extractors
        self.attn = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)) # υπολογίζουμε attention score
        self.drop = nn.Dropout(0.3) # για να μην κάνει Overfitting
        self.organ = nn.Linear(512, 13) # εδώ προβλέπει το όργανο
        self.tumor = nn.Linear(512, 2) # εδώ την κατάσταση (δηλ αν έχει όγκο ή όχι)
    def forward(self, x):
        B, S, C, H, W = x.shape # input tensor με Batch size, Slices, Channel, Height & Width
        x = x.view(B * S, C, H, W) # τα βάζει μαζί, οπότε τώρα νομίζει ότι όλα είναι μεμονομένες 2D εικόνες
        f = self.backbone(x).view(B, S, 512) # το resnet18 βλέπει τις εικόνες και εξάγει 512 χαρακτηριστικά
        w = torch.softmax(self.attn(f), dim=1) # βάζουμε τα βάρη
        f = (f * w).sum(1) # κάνει ένα εννιαίο διάνυσμα για να βρει τις πιο σημαντικές εικόνες
        f = self.drop(f) # το dropout για να μην έχουμε Overfitting
        return self.organ(f), self.tumor(f) # μας λέει τις τελικές προβλέψεις

# Εκπαίδευση ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train(model, loader, device):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4) # βελτιστοποίηση με τον κύριο Adam (ορίζει ρυθμό μάθησης) για να ανανεώνει τα βάρη
    loss_fn = nn.CrossEntropyLoss() # έβαλα cross entropy για συνάρτηση κόστους επειδή είναι για ταξινόμηση
    model.to(device) # στέλνει το μοντέλο στην cpu ή στο cuda
    # αυτά τα θέλουμε για τα confusion matrix μετά
    all_preds_o, all_true_o = [], []
    all_preds_t, all_true_t = [], []
    # εκπαιδεύεται σε τρεις κύκλους (εποχές)
    for epoch in range(3):
        model.train() # λέμε στο μοντέλο να κάτσει να διαβάσει (ενεργοποιεί dropout)
        total = 0 # δεν έχουμε Loss ακόμα
        for x, oy, ty in tqdm(loader): # για να έχουμε και την ωραία μπάρα στην κονσόλα
            x, oy, ty = x.to(device), oy.to(device), ty.to(device) # στέλνουμε εικόνες και πραγματικές ταμπέλες όγκων κ οργάνων
            opt.zero_grad() # κάνουμε τα Gradient μηδέν για να μην μαζεύονται από τα προηγούμενα epoch
            o, t = model(x) # προβλέπει όργανο και κατάσταση/όγκο
            loss = loss_fn(o, oy) + loss_fn(t, ty) # προσθέτει τα σφάλματα στους όγκους και τα σφάλματα στα όργανα
            loss.backward() # back propagation για να αλλάξει τα βάρη και να έχει καλύτερα αποτελέσματα στην επόμενη επανάληψη
            opt.step() # ανανεώνει τα βάρη
            total += loss.item() # προσθέτει τα σφάλματα στο συνολικό Loss κάθε εποχής
            all_preds_o.extend(o.argmax(1).cpu().numpy()) # κάνει τις προβλέψεις να είναι στην κλάση με την μεγαλύτερη πιθανότητα
            all_true_o.extend(oy.cpu().numpy()) # κάνει τα αποτελέσματα πίνακες Numpy για να μπορεί να τα αποθηκεύσει και σε λίστα Python
            all_preds_t.extend(t.argmax(1).cpu().numpy()) # τα ίδια και για την κατάσταση (όγκος ή καθαρό)
            all_true_t.extend(ty.cpu().numpy())
        print("epoch", epoch, total / len(loader)) # τυπώνουμε κιόλας
    return all_true_o, all_preds_o, all_true_t, all_preds_t # αυτά τα θέλουμε για το επόμενο!

# Αξιολόγηση ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def evaluate(model, loader, device):
    model.eval() # λέμε στο μοντέλο ότι του βάζουμε διαγώνισμα (απενεργοποιεί dropout)
    # κρατάμε εδώ αυτά
    all_preds_o, all_true_o = [], []
    all_preds_t, all_true_t = [], []
    with torch.no_grad(): # δεν υπολογίζει gradient αφού δεν χρειάζονται εδώ
        for x, oy, ty in tqdm(loader): # και εδώ progress bar, έτσι για το γούρι
            x, oy, ty = x.to(device), oy.to(device), ty.to(device) # στέλνει αυτά που χρειάζεται όπως πριν
            o, t = model(x) # προβέπει πάλι όργανο και κατάσταση
            # επιλέγει την κλάση με την μεγαλύτερη πιθανότητα και μεταφέρει από την cpu για να προσθέσουμε στα προηγούμενα
            all_preds_o.extend(o.argmax(1).cpu().numpy())
            all_true_o.extend(oy.cpu().numpy())
            all_preds_t.extend(t.argmax(1).cpu().numpy())
            all_true_t.extend(ty.cpu().numpy())
    # τυπώνουμε τα αποτελέσματα μέσω sklearn (το zero division = 0 για να μην κρασάρει άμα κάποια κλάση δεν την βρήκε καθόλου)
    print("\n˗ˋˏ TEST ORGAN REPORT ˎˊ˗")
    print(classification_report(all_true_o, all_preds_o, zero_division=0))
    print("\n˗ˋˏ TEST TUMOR REPORT ˎˊ˗")
    print(classification_report(all_true_t, all_preds_t, zero_division=0))
    organ_labels = list(VolumeDataset(loader.dataset.df).organ_map.keys()) # παίρνει τις ταμπέλες για τα όργανα για να τις βάλει στην εικονίτσα
    # confusion matrix για όργανο και κατάσταση μέσω matplotlib :)
    plot_cm(all_true_o, all_preds_o, organ_labels, "TEST ~ Organ Confusion Matrix")
    plot_cm(all_true_t, all_preds_t, ["no_tumor", "tumor"], "TEST ~ Tumor Confusion Matrix")

# Για τα plots ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_cm(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))) # για να δείξει οπωσδήποτε όλα τα όργανα
    plt.figure(figsize=(10, 8)) # μέγεθος για το παράθυρο του Plot
    plt.imshow(cm, cmap="Blues") # με μπλε! :)
    plt.title(title) # τίτλος
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right") # τα γράμματα εδώ στο πλάι για να μην πέφτουν το ένα πάνω στο άλλο
    plt.yticks(range(len(labels)), labels) # εδώ δεν χρειάζεται γιατί είναι το ένα κάτω από το άλλο
    for i in range(cm.shape[0]): # εμφωλευμένες for για γραμμές και στήλες
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8) # για να βλέπουμε τους αριθμούς μέσα στα κουτάκια
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout() # για να μην κόβει τα γράμματα
    plt.show() # έτοιμο!

# η main μας ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__": # πάμεεεε!
    device = get_device() # για άμα έχει cuda
    wandb.init(project="idc-multitask", mode="offline") # κρατάει Logs μέσω wandb, offline γιατί δεν έχω λογαριασμό
    # οι συλλογές που διαβάζει στην εκπαίδευση
    train_sets = [
        "4d_lung", "breast_diagnosis", "colorectal_liver_metastases",
        "cptac_cm", "remind", "cmb_ov", "covid_19_ar", "prostate_mri",
        "pancreas_ct", "prostatex", "spine_mets_ct_seg", "acrin_flt_breast"
    ]
    # οι συλλογές που μαντεύει στο τεστ
    test_sets = ["covid_19_ny_sbu", "ctpred_sunitinib_pannet"]
    # τα κάνει εννοιαία dataframe
    train_df = build_multi(train_sets)
    test_df = build_multi(test_sets)
    # φορτώνουμε ότι χρειαζόμαστε για την εκπαίδευση
    train_loader = DataLoader(
        VolumeDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True, # τα ανακατεύει για να μην κλέβει
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    # το ίδιο και για το τεστ
    test_loader = DataLoader(
        VolumeDataset(test_df),
        batch_size=BATCH_SIZE,
        shuffle=False, # εδώ δεν χρειάζεται
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    model = Model() # φωνάζουμε το μοντέλο μας
    y_o, p_o, y_t, p_t = train(model, train_loader, device) # κάνει την εκπαίδευση του όπως είπαμε πάνω
    evaluate(model, test_loader, device) # κάνει και την αξιολόγηση
    # τα αποτελέσματα με εικονίτσες :)
    plot_cm(y_o, p_o, list(VolumeDataset(train_df).organ_map.keys()), "Organ Confusion Matrix")
    plot_cm(y_t, p_t, ["no_tumor", "tumor"], "Tumor Confusion Matrix")
    # τέλος!

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Αφιερωμένο στον Πάμφιλο, που τώρα που μεγάλωσε, μπλατσανάει στα βαθιά και όχι στα ρηχά
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Σε αγαπάμε θείο μου <3
