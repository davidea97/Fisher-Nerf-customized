CONFIG=$1
DATADIR=../data/versioned_data/


# HM3D scenes
# SCENES=("00010-DBjEcHFg4oq" "00246-mscxX4KEBcB" "00285-QKGMrurUVbk" "00033-oPj9qMxrDEa"  "00051-CETmJJqkhcK")

# Gibson scenes
DATASET="gibson"
DATASET_SPLIT="train"
# SCENES=("Greigsville" "Denmark" "Cantwell" "Eudora" "Pablo" "Ribera" "Swormville" "Eastville" "Elmira")
SCENES=("Greigsville")


# MP3D scenes
# DATASET="MP3D"
# DATASET_SPLIT="train"
# SCENES=("GdvgFV5R1Z5" "gZ6f7yhEvPG" "HxpKQynjfin" "pLe4wQe7qrG" "YmJkqBEsHnH")
# SCENES=("GdvgFV5R1Z5")

# Test
# SCENES=("00800-TEEsavR23oF")


for scene in ${SCENES[@]}
do
    python main.py --name test_pointnav_exp \
                    --ensemble_dir ckpt/ \
                    --slam_config ${CONFIG} \
                    --root_path ${DATADIR} \
                    --log_dir logs/ \
                    --scenes_list ${scene} \
                    --gpu_capacity 1 \
                    --test_set v1 \
                    --dataset ${DATASET}
done