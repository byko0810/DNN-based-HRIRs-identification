# DNN-based-HRIRs-identifications
DNN-based HRIRs identifications using a continuous rotating speaker array

Download binaural recording files from Zenodo
https://doi.org/10.5281/zenodo.15004594
and upload the files on the 'data' folder

## HRIR identification using a continuous rotating speaker array
![Image](https://github.com/user-attachments/assets/b2931950-27e6-4ed6-917a-97a8dcfb7dbf)

## Abstract
Conventional static measurement of head-related impulse responses (HRIRs) is time-consuming due to the need for repositioning a speaker array for each azimuth angle. Dynamic approaches using analytical models with a continuously rotating speaker array have been proposed, but their accuracy is significantly reduced at high rotational speeds. To address this limitation, we propose a DNN-based HRIRs identification using sequence-to-sequence learning. The proposed DNN model incorporates fully connected (FC) networks to effectively capture HRIR transitions and includes reset and update gates to identify HRIRs over a whole sequence. The model updates the HRIRs vector coefficients based on the gradient of the instantaneous square error (ISE). Additionally, we introduce a learnable normalization process based on the speaker excitation signals to stabilize the gradient scale of ISE across time. A training scheme, referred to as whole-sequence updating and optimization scheme, is also introduced to prevent overfitting. We evaluated the proposed method through simulations and experiments. Simulation results using the FABIAN database show that the proposed method outperforms previous analytic models, achieving over 7 dB improvement in normalized misalignment (NM) and maintaining log spectral distortion (LSD) below 2 dB at a rotational speed of 45°/s. Experimental results with a custom-built speaker array confirm that the proposed method successfully preserved accurate sound localization cues, consistent with those from static measurement.

For more details, please see: ["DNN based HRIRs Identification with a Continuously Rotating Speaker Array"](https://arxiv.org/abs/2504.14817) [Byeong-Yun Ko](https://scholar.google.com/citations?user=iaquQiAAAAAJ&hl=ko&oi=sra) [Deokki Min](https://scholar.google.com/citations?hl=ko&user=Wm7WmcIAAAAJ) [Hyeonuk Nam](https://scholar.google.com/citations?hl=ko&user=rCN5da8AAAAJ) [Yong-Hwa Park](https://scholar.google.com/citations?hl=ko&user=LtZKH8wAAAAJ) arXiv 2025.
