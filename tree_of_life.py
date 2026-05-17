# οι βιβλιοθήκες / the libraries ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from dataclasses import dataclass # για να αποθηκεύουμε δεδομένα με δική μας δομή / so we can save data in our own format
import csv # για να μπορεί να διαβάσει το φύλλο με τις θεραπείες / so it can read the spreadsheet with the treatments
from typing import Optional, Dict, Any # για πιο καλή οργάνωση / for better organisation

# το "μοντέλο" του ασθενή / the patient "model" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@dataclass
class Patient:
    # γενικά / generals
    sex: str # γένος (ανατομικά, όχι φύλλο!) / sex (anatomically, not gender!)
    age: int # ηλικία / age
    cancer_type: str # σε ποιό όργανο είναι ο καρκίνος / which organ the cancer is at
    cancer_stage: str # και σε ποιό στάδιο / and which stage
    race: Optional[str] = None # στην έρευνα είχε και την φυλή (αν και δεν νομίζω ότι χρησιμεύει κάπου) / in the paper it also had race (though I don't think it's useful anywhere)
    # βιολογικές λεπτομέριες / biological details
    hormone_receptor_positive: Optional[bool] = None
    her2_positive: Optional[bool] = None
    braf_mutated: Optional[bool] = None
    low_risk: Optional[bool] = None
    # ανατομικές λεπτομέριες / anatomical details
    anatomy_notes: Optional[str] = None
    # ιατρικό ιστορικό και γενικά παράγοντες κινδύνου / medical history and comorbidities
    autoimmune_disease: Optional[bool] = None
    autoimmune_type: Optional[str] = None
    organ_transplant: Optional[bool] = None
    chronic_steroid_use: Optional[bool] = None
    cardiovascular_disease: Optional[bool] = None
    lung_disease: Optional[bool] = None
    renal_impairment: Optional[bool] = None
    liver_impairment: Optional[bool] = None
    pregnant: Optional[bool] = None
    smoker: Optional[bool] = None

# άλλα χρήσιμα πράγματα / other useful things ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# επειδή η κάθε τιμή που αντιστοιχεί σε ένα κλειδί πρέπει να είναι μοναδική / because each value that corresponds to a key has to be unique
def unique(items):
    return list(dict.fromkeys(items))

# για να καταλαβαίνει τα στάδια / so it can understand the stages
def normalize_stage(stage: str) -> str:
    stage = stage.strip().upper()
    mapping = {"1": "I", "2": "II", "3": "III", "4": "IV"}
    return mapping.get(stage, stage)

# για να καταλαβαίνει τα ναι και όχι / so it can understand the yes and no
def ask_boolean(question: str) -> bool:
    while True:
        answer = input(question + " (y/n): ").strip().lower()
        if answer in ["y", "yes"]: return True
        if answer in ["n", "no"]: return False
        print("Please enter y/n.")

# τα φίλτρα για το ιατρικό ιστορικό κ τους παράγοντες κινδύνου / the filters for comorbidity and medical history ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def apply_comorbidity_filters(
    patient: Patient,
    recommendations: list, # δίνει προτάσεις / gives recommendations
    warnings: list # και προειδοποιήσεις / and warnings!
):
    # για αυτοάνοσα / for autoimmunes
    if patient.autoimmune_disease:
        warnings.append("Pre-existing autoimmune disease may increase risk of immune-related adverse events.")
        warnings.append("Requires multidisciplinary evaluation (oncology/rheumatology/immunology).")
        # για έξτρα ασφάλεια / extra safety
        filtered = []
        for treatment in recommendations:
            if "immunotherapy" in treatment.lower():
                warnings.append("Immunotherapy removed due to autoimmune risk.")
                continue
            filtered.append(treatment)
        recommendations = filtered

    # για μεταμοσχεύσεις / for transplants
    if patient.organ_transplant:
        warnings.append("Checkpoint inhibitors may increase risk of graft rejection.")

    # για χρόνια χρήση στεροειδών / for chronic use of steroids
    if patient.chronic_steroid_use:
        warnings.append("Chronic corticosteroid use may reduce immunotherapy efficacy.")

    # για ασθένειες του πνεύμονα / for lung diseases
    if patient.lung_disease:
        warnings.append("Underlying lung disease may increase risk of pneumonitis during radiation/immunotherapy.")

    # για καρδιομυοπάθειες / for cardiovascular disaeses
    if patient.cardiovascular_disease:
        warnings.append("Cardiovascular disease may increase risk of cardiotoxicity.")

    # για νεφρική ανεπάρκεια / for renal impairment
    if patient.renal_impairment:
        warnings.append("Renal impairment may require dose adjustment for nephrotoxic chemotherapy agents.")

    # για ηπατική ανεπάρκεια / for liver impairment
    if patient.liver_impairment:
        warnings.append("Liver dysfunction may affect metabolism of systemic therapies.")

    # για εγκυμοσύνες / for pregnancies
    if patient.pregnant:
        warnings.append("Certain chemotherapy, radiation, and targeted therapies may be contraindicated in pregnancy.")

    # για καπνιστές / for smokers
    if patient.smoker:
        warnings.append("Smoking history may worsen pulmonary toxicity and treatment outcomes.")
    # μας τα φέρνει / brings them to us
    return unique(recommendations), unique(warnings)

# η βάση δεδομένων! / the drug database! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# για να φορτώσουμε τα δεδομένα / to load the data
def load_drugs_by_primary_site(csv_path="drugs.csv"): # πρέπει το drugs.csv να είναι στον ίδιο φάκελο με αυτό το αρχείο! / drugs.csv has to be in the same folder as this file!
    grouped = {} # θα τα κρατήσουμε εδώ / we'll keep them here
    try:
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get("Do not code", "").strip().lower() == "yes": continue # παραλείπουμε κενές ή ημιτελείς θεραπείες / we skip blank or partial entries
                primary_site = row.get("Primary Site", "Unknown").strip()
                drug_name = row.get("Name", "Unknown").strip()
                category = row.get("Category", "N/A").strip()
                sub_category = row.get("Sub-category", "N/A").strip()
                histology = row.get("Histology", "N/A").strip()
                remarks = row.get("Remarks", "").strip()
                abbreviation = row.get("Abbreviation", "").strip()
                nsc_number = row.get("NSC number", "").strip()
                evs_id = row.get("EVS ID", "").strip()
                alternate_name = row.get("Alternate Name", "").strip()
                if primary_site not in grouped: grouped[primary_site] = []
                grouped[primary_site].append({"drug": drug_name, "category": category, "sub_category": sub_category, "histology": histology, 
                "remarks": remarks, "abbreviation": abbreviation, "nsc_number": nsc_number, "evs_id": evs_id, "alternate_name": alternate_name})
    except FileNotFoundError:
        print("Warning: drugs.csv not found! Reference analytics disabled.") # άμα δεν βρίσκει το αρχείο / if it can't find the file
    return grouped

# για να τυπώσουμε τα κατ'αλληλα φάρμακα / to print the right drugs
def display_drug_analytics(
    drug_database,
    cancer_type
):
    print("\n•.° - ₊˚ʚ ɞ˚₊ - °.• Analytical oncology reference data! •.° - ₊˚ʚ ɞ˚₊ - °.•")
    normalized = cancer_type.lower()
    matches = []
    for primary_site, drugs in drug_database.items():
        if normalized in primary_site.lower(): matches.extend(drugs)
    if not matches:
        print(f"No reference entries found for: {cancer_type}") # αν δεν υπάρχει κάτι / if there is nothing
        return
    print(f"\nPrimary Site Match: {cancer_type.title()}")
    print("-" * 60)
    for idx, drug in enumerate(matches, start=1):
        print(f"\n[{idx}] {drug['drug']}")
        print(f"Category: {drug['category']}")
        print(f"Sub-category: {drug['sub_category']}")
        print(f"Histology: {drug['histology']}")
        if drug["abbreviation"]: print(f"Abbreviation: {drug['abbreviation']}")
        if drug["nsc_number"]: print(f"NSC Number: {drug['nsc_number']}")
        if drug["evs_id"]: print(f"EVS ID: {drug['evs_id']}")
        if drug["alternate_name"]: print(f"Alternate Name: {drug['alternate_name']}")
        if drug["remarks"]: print(f"Remarks: {drug['remarks']}")

# decision tree! :) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class CancerTreatmentDecisionTree:
    def evaluate(self, patient: Patient) -> Dict[str, Any]:
        if patient.age < 18: return self._pediatric_branch(patient) # τα παιδάκι στην παιδιατρική / the kiddies go to pediatrics dpt
        cancer = patient.cancer_type.lower()
        if cancer == "breast": return self._breast_branch(patient)
        elif cancer == "prostate": return self._prostate_branch(patient)
        elif cancer == "lung": return self._lung_branch(patient)
        elif cancer == "colorectal": return self._colorectal_branch(patient)
        elif cancer == "melanoma": return self._melanoma_branch(patient)
        elif cancer == "thyroid": return self._thyroid_branch(patient)
        elif cancer == "testicular": return self._testicular_branch(patient)
        # και για άμα είναι κάτι που δεν ήταν στην έρευνα / and if it's something that wasn't in the research
        else: return {"status": "unsupported", "message": (f"Cancer type '{patient.cancer_type}' not yet implemented.")}

    # για τα παιδάκια! / for the kiddies!
    def _pediatric_branch(self, patient: Patient):
        recommendations = ["Pediatric oncology referral", "Age-specific chemotherapy protocols", "Consider surgery/radiation depending on tumor type", "Fertility preservation counseling"]
        warnings = []
        recommendations, warnings = apply_comorbidity_filters(patient, recommendations, warnings)
        return {
            "cancer_type": "Pediatric/Adolescent Oncology",
            "recommended_treatments": recommendations,
            "survivorship_risks": ["Cardiotoxicity", "Neurocognitive effects", "Endocrine dysfunction", "Growth abnormalities", "Secondary malignancies"], "warnings": warnings}

    # στήθος / breast
    def _breast_branch(self, patient: Patient):
        recommendations = []
        if patient.cancer_stage in ["I", "II", "LOCALIZED", "EARLY"]:
            recommendations.extend(["Breast-conserving surgery", "Radiation therapy"])
            if patient.hormone_receptor_positive: recommendations.append("Endocrine therapy")
            if patient.her2_positive: recommendations.append("HER2-targeted therapy")
        else: recommendations.extend(["Systemic therapy", "Chemotherapy", "Targeted immunotherapy"])
        recommendations = unique(recommendations)
        warnings = []
        recommendations, warnings = apply_comorbidity_filters(patient, recommendations, warnings)
        return {"cancer_type": "Breast Cancer", "recommended_treatments": recommendations, "survivorship_risks": ["Lymphedema", "Cardiotoxicity", "Peripheral neuropathy", "Cognitive impairment"], "warnings": warnings}

    # προστάτης / prostate
    def _prostate_branch(self, patient: Patient):
        recommendations = []
        if patient.low_risk: recommendations.append("Active surveillance")
        if patient.cancer_stage in ["I", "II", "LOCALIZED"]:
            recommendations.extend(["Radical prostatectomy", "Radiation therapy"])
        else: recommendations.extend(["Hormonal therapy", "Systemic therapy"])
        if patient.age >= 75: recommendations.append("Conservative management")
        recommendations = unique(recommendations)
        warnings = []
        recommendations, warnings = apply_comorbidity_filters(patient, recommendations, warnings)
        return {"cancer_type": "Prostate Cancer", "recommended_treatments": recommendations, "survivorship_risks": ["Urinary incontinence", "Sexual dysfunction", "Fatigue"], "warnings": warnings}

    # πνευμόνια / lungs
    def _lung_branch(self, patient: Patient):
        recommendations = []
        notes = []
        if patient.cancer_stage in ["I", "II"]:
            recommendations.append("Surgical resection")
            if (patient.race and patient.race.lower() == "black"): notes.append("Observed surgery rate: 47% for Black patients")
            elif (patient.race and patient.race.lower() == "white"): notes.append("Observed surgery rate: 52% for White patients")
        else: recommendations.extend(["Chemotherapy", "Immunotherapy", "Targeted therapy", "Radiation therapy"])
        recommendations = unique(recommendations)
        warnings = []
        recommendations, warnings = apply_comorbidity_filters(patient, recommendations, warnings)
        return {"cancer_type": "Lung Cancer", "recommended_treatments": recommendations, "notes": notes, "survivorship_risks": ["Pulmonary dysfunction", "Fatigue", "Anxiety/depression"], "warnings": warnings}

    # παχύ έντερο / colorectal
    def _colorectal_branch(self, patient: Patient):
        recommendations = []
        notes = []
        if patient.cancer_stage in ["I", "II", "III"]: recommendations.append("Surgery")
        if patient.cancer_stage in ["II", "III", "IV"]: recommendations.append("Chemotherapy")
        if patient.cancer_stage in ["III", "IV", "LOCALLY ADVANCED"]: recommendations.extend(["Chemoradiation", "Rectal surgery", "Adjuvant chemotherapy"])
        if (patient.race and patient.race.lower() == "black"): notes.append("Observed Stage I rectal surgery rate: 39%")
        elif (patient.race and patient.race.lower() == "white"): notes.append("Observed Stage I rectal surgery rate: 64%")
        recommendations = unique(recommendations)
        warnings = []
        recommendations, warnings = apply_comorbidity_filters(patient, recommendations, warnings)
        return {"cancer_type": "Colorectal Cancer", "recommended_treatments": recommendations, "notes": notes, "survivorship_risks": ["Neuropathy", "Bowel dysfunction", "Sexual dysfunction"], "warnings": warnings}

    # Μελάνωμα / melanoma
    def _melanoma_branch(self, patient: Patient):
        recommendations = []
        if patient.cancer_stage in ["I", "II", "LOCALIZED"]: recommendations.append("Surgical excision")
        else:
            recommendations.append("Immune checkpoint inhibitors")
            if patient.braf_mutated: recommendations.append("BRAF-targeted therapy")
        recommendations = unique(recommendations)
        warnings = []
        recommendations, warnings = apply_comorbidity_filters(patient, recommendations, warnings)
        return {"cancer_type": "Melanoma", "recommended_treatments": recommendations, "survivorship_risks": ["Recurrence anxiety", "Immune-related toxicity"], "warnings": warnings}

    # θυροειδής / thyroid
    # ("θυρεοτηλέφωνο!!!!")
    def _thyroid_branch(self, patient: Patient):
        recommendations = ["Thyroidectomy"]
        if patient.cancer_stage not in ["I", "LOCALIZED"]:
            recommendations.append("Radioactive iodine")
        recommendations.append("Thyroid hormone suppression therapy")
        recommendations = unique(recommendations)
        warnings = []
        recommendations, warnings = apply_comorbidity_filters(patient, recommendations, warnings)
        return {"cancer_type": "Thyroid Cancer", "recommended_treatments": recommendations, "survivorship_risks": ["Endocrine dysfunction", "Fatigue", "Voice/swallowing complications"], "warnings": warnings}

    # όρχεις / testicles
    # (και όπως είπε και ο κύριος Κουτσούμπας, τώρα ξέρουμε από τι είναι ο καπαμάς)
    def _testicular_branch(self, patient: Patient):
        recommendations = []
        statistics = []
        if patient.cancer_stage == "I":
            recommendations.append("Orchiectomy")
            statistics.append("Stage I seminoma orchiectomy alone: 78%")
        elif patient.cancer_stage == "II":
            recommendations.extend(["Chemotherapy", "Radiation therapy"])
            statistics.extend(["Stage II seminoma chemotherapy: 66%", "Stage II seminoma radiation: 19%"])
        else:
            recommendations.extend(["Surgery", "Chemotherapy"])
            statistics.append("Late-stage seminoma surgery + chemotherapy without radiation: 68%")
        recommendations = unique(recommendations)
        warnings = []
        recommendations, warnings = apply_comorbidity_filters(patient, recommendations, warnings)
        return {"cancer_type": "Testicular Cancer", "recommended_treatments": recommendations, "statistics": statistics, "survivorship_risks": ["Fertility issues", "Neuropathy", "Cardiovascular risk"], "warnings": warnings}

# Υποστηριζόμενες μορφές καρκίνου (αυτές που είχε στην έρευνα) / Supported cancer types (those that were in the research paper) ~~~~~~~~~~~~~~~~~~~

SUPPORTED_CANCERS = {
    "1": "breast",
    "2": "prostate",
    "3": "lung",
    "4": "colorectal",
    "5": "melanoma",
    "6": "thyroid",
    "7": "testicular"
}

# τα της κονσόλας / console stuff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# για να ξέρουν οι χρήστες ποιά μπορούν να διαλ'εξουν / so the users will know what they can choose from
def print_supported_cancers():
    print("\nSupported Cancer Types")
    print("˚₊‧꒰ა ☆ ໒꒱ ‧₊˚˚₊‧꒰ა ☆ ໒꒱ ‧₊˚")
    for key, value in SUPPORTED_CANCERS.items():
        print(f"{key}. {value.title()}")

# για να μην γράφουν ότι να ναι / so they won't type whatever
def get_valid_age():
    while True:
        try:
            age = int(input("Age: "))
            if age < 0 or age > 120:
                print("Please enter a realistic age.")
                continue
            return age
        except ValueError:
            print("Invalid age. Please enter a number.")

# για να παίρνει μόνο αυτές τις επιλογές / so it can take only these options
def get_valid_sex():
    while True:
        sex = input("Biological sex (male/female/other): ").strip().lower()
        if sex in ["male", "female", "other"]: return sex
        print("Please enter male/female/other.")

# έλεγχος για το γένος / check for the sex
def get_cancer_choice(sex):
    while True:
        print_supported_cancers()
        choice = input("\nSelect cancer type number: ").strip()
        if choice not in SUPPORTED_CANCERS:
            print("Invalid selection.") # αν είναι κάτι από αυτά που έχουμε / if it's not one of what we have
            continue
        cancer = SUPPORTED_CANCERS[choice]
        if sex == "male" and cancer == "breast": # γίνεται! / is possible!
            print("Male breast cancer is rare but possible!")
        if (sex == "female" and cancer in ["prostate", "testicular"]): # επειδή δεν έχουν απ αυτά / because they don't have those
            print(f"{cancer.title()} cancer is incompatible with typical female anatomy.")
            continue
        return cancer

# για να πάει τα παιδάκια στον σωστό κλάδο / to take the kiddies to the correct branch
def pediatric_redirect(age):
    if age < 18:
        print("\nThis is a pediatric/adolescent case!")
        print("Recommend pediatric oncology pathway.")

# main! :D ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    print("Cancer Treatment Decision Tree Prototype")
    print("₊˚⊹ ₊˚⊹ ₊˚⊹ ₊˚⊹ ₊˚⊹ ₊˚⊹ ₊˚⊹ ₊˚⊹ ₊˚⊹ ₊˚⊹") # will probably remove the fun symbols later, had to do something to prevent crying
    drug_database = load_drugs_by_primary_site()
    sex = get_valid_sex()
    age = get_valid_age()
    pediatric_redirect(age)
    anatomy_notes = None
    if sex == "other": anatomy_notes = input("Relevant anatomy / hormone exposure notes: ").strip()
    cancer_type = get_cancer_choice(sex)
    cancer_stage = normalize_stage(input("Cancer stage (I, II, III, IV, localized, early, advanced): "))
    race = input("Race (optional): ").strip().lower()
    if race == "": race = None
    patient = Patient(sex=sex, age=age, cancer_type=cancer_type, cancer_stage=cancer_stage, race=race, anatomy_notes=anatomy_notes)
    # ιατρικό ιστορικό κ παράγοντες κινδύνου / comorbidities adn med history
    patient.autoimmune_disease = ask_boolean("History of autoimmune disease?")
    if patient.autoimmune_disease: patient.autoimmune_type = input("Autoimmune disease type: ").strip()
    patient.organ_transplant = ask_boolean("History of organ transplant?")
    patient.chronic_steroid_use = ask_boolean("Chronic corticosteroid use?")
    patient.cardiovascular_disease = ask_boolean("Cardiovascular disease?")
    patient.lung_disease = ask_boolean("Chronic lung disease?")
    patient.renal_impairment = ask_boolean("Renal impairment?")
    patient.liver_impairment = ask_boolean("Liver impairment?")
    patient.smoker = ask_boolean("Smoking history?")
    if sex != "male": patient.pregnant = ask_boolean("Pregnancy present?")
    # dynamic questions!
    if cancer_type == "breast":
        patient.hormone_receptor_positive = ask_boolean("Hormone receptor positive?")
        patient.her2_positive = ask_boolean("HER2 positive?")
    elif cancer_type == "prostate": patient.low_risk = ask_boolean("Low-risk disease?")
    elif cancer_type == "melanoma": patient.braf_mutated = ask_boolean("BRAF mutation present?")
    # αξιολόγηση / evaluationm
    tree = CancerTreatmentDecisionTree()
    result = tree.evaluate(patient)
    # αναλυτικά όλα τα φάρμακα που είναι οκ για την περίπτωση / analytics for all the meds that are ok for the case
    see_drugs = ask_boolean("Would you like to view analytical oncology reference data?") # μόνο άμα θέλουν / only if they want
    if see_drugs: display_drug_analytics(drug_database, cancer_type)
    # τυπώνουμε / we print!
    print("\n𓂃˖ ࣪⊹ Decision Results ⊹ ࣪˖𓂃")
    for key, value in result.items():
        print(f"\n{key.upper()}:")
        if isinstance(value, list):
            if len(value) == 0: print(" - None")
            else:
                for item in value:
                    print(f" - {item}")
        else: print(value)
    # και μια ενημέρωση εδώ για καλό και για κακό / and a disclaimer here just in case
    print("\nDISCLAIMER: This is not professional medical software and is only meant for educational and research purposes. Please consult a medical professional!")

# να είστε όλοι καλά! :) / may you all be well! :) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
