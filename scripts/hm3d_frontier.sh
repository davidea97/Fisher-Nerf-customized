CONFIG=$1
DATADIR=../data/versioned_data/

# SCENES=("DBjEcHFg4oq" "mscxX4KEBcB" "QKGMrurUVbk" "oPj9qMxrDEa"  "CETmJJqkhcK")
# SCENES=("00010-DBjEcHFg4oq" "00246-mscxX4KEBcB" "00285-QKGMrurUVbk" "00033-oPj9qMxrDEa"  "00051-CETmJJqkhcK")
SCENES=("00800-TEEsavR23oF")
# SCENES=("00814-p53SfW6mjZe" "00832-qyAac8rV8Zk" "00879-7Ukhou1GxYi" "00891-cvZr5TUy5C5"  "00898-8CRYizAb6yd")


for scene in ${SCENES[@]}
do
    python main_navigation.py --name test_pointnav_exp \
                    --ensemble_dir ckpt/ \
                    --slam_config ${CONFIG} \
                    --root_path ${DATADIR} \
                    --log_dir logs/ \
                    --scenes_list ${scene} \
                    --gpu_capacity 1 \
                    --test_set v1 
done