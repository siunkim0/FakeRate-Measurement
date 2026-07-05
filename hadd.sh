cd /data6/Users/snuintern2/fake/stack
aa="2016_2"
mkdir ${aa}


BASE_PATH="/data6/Users/snuintern2/fake/2016preVFP_2"

BASE_PATH1="/data6/Users/snuintern2/fake/2016postVFP_2" 

BASE_PATH2="/data6/Users/snuintern2/fake/stack/${aa}" 

#hadd -f -j 8 ${BASE_PATH}/Muon.root "${input_files_DATA[@]}"
hadd -f -j 8 ${BASE_PATH}/QCD.root ${BASE_PATH}/QCD_*.root

hadd -f -j 8 ${BASE_PATH1}/QCD.root ${BASE_PATH1}/QCD_*.root

hadd -f -j 8 ${BASE_PATH2}/QCD.root ${BASE_PATH}/QCD.root ${BASE_PATH1}/QCD.root

#hadd -f -j 8 DoubleMuon.root DoubleMuon_*.root
#hadd -f -j 8 DY.root DYJets*.root