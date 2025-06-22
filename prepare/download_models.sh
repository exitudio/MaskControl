# rm -rf checkpoints
mkdir -p checkpoints
cd checkpoints
mkdir -p t2m

cd t2m 
echo -e "Downloading pretrained models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0/view?usp=sharing

echo -e "Unzipping humanml3d_models.zip"
unzip humanml3d_models.zip

echo -e "Cleaning humanml3d_models.zip"
rm humanml3d_models.zip


gdown --fuzzy https://drive.google.com/file/d/1YOUWPsF1c9YCDFUoMae8Cd69TpC9eW1e/view?usp=sharing
echo -e "Unzipping ControlMM pelvis only"
unzip z2024-08-23-01-27-51_CtrlNet_randCond1-196_l1.1XEnt.9TTT__fixRandCond.zip
rm z2024-08-23-01-27-51_CtrlNet_randCond1-196_l1.1XEnt.9TTT__fixRandCond.zip

gdown --fuzzy https://drive.google.com/file/d/15d-xG2ECs4q8rFO9d6ctF4LGJmQH6GGV/view?usp=sharing
echo -e "Unzipping ControlMM all joints"
unzip z2024-08-27-21-07-55_CtrlNet_randCond1-196_l1.5XEnt.5TTT__cross.zip
rm z2024-08-27-21-07-55_CtrlNet_randCond1-196_l1.5XEnt.5TTT__cross.zip

cd ../../

echo -e "Downloading done!"