# the-thesis-that-gave-me-tendonitis
My final thesis for university, and personal research on cancer.

DISCLAIMER : I am not a doctor or any sort of medical professional! I am merely a computer science student hoping to maybe help someone smarter than myself with their own research. The link to my research paper will be included here once published.

Pamphilos.py is a CNN model designed to detect whether an x-ray scan image contains a tumor or not. The model uses ResNet18 as a backbone, trains on samples provided by the public IDC database of the US National Institute for Cancer, contains several safety mechanisms, and creates informational graphs via external libraries.

Tree_of_life.py is a very primitive Decision Tree algorithm, designed to list the best available treatments for a specific cancer patient, based on user responses to several questions. The csv data used is provided by NIC's SEER*Rx Interactive Antineoplastic Drugs Database, and the decisional structure is based on the following American Cancer Society research journal, and two NIH research articles :

1) Wagle NS, Nogueira L, Devasia TP, et al. Cancer treatment and survivorship statistics, 2025. CA Cancer J Clin. 2025;75(4):308-340. doi:10.3322/caac.70011

2) Florou, V., Puri, S., Garrido-Laguna, I., & Wilky, B. A. (2021). Considerations for immunotherapy in patients with cancer and comorbid immune dysfunction. Annals of translational medicine, 9(12), 1035. https://doi.org/10.21037/atm-20-5207

3) Kehl, K. L., Yang, S., Awad, M. M., Palmer, N., Kohane, I. S., & Schrag, D. (2019). Pre-existing autoimmune disease and the risk of immune-related adverse events among patients receiving checkpoint inhibitors for cancer. Cancer immunology, immunotherapy : CII, 68(6), 917–926. https://doi.org/10.1007/s00262-019-02321-z

Feel free to use this in your own research and modify as you see fit (I've put most of the important things in variables under the settings section, so they'll be easier to find and change). Credit would be deeply appreciated!

Dedicated to my uncle, Pamphilos :)
