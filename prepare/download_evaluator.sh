cd checkpoints

cd t2m 
echo -e "Downloading evaluation models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/19C_eiEr0kMGlYVJy_yFL6_Dhk3RvmwhM/view?usp=sharing
echo -e "Unzipping humanml3d_evaluator.zip"
unzip humanml3d_evaluator.zip

echo -e "Clearning humanml3d_evaluator.zip"
rm humanml3d_evaluator.zip


cd ../../

echo -e "Downloading done!"