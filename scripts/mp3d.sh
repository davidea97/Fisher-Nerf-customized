CONFIG=$1
DATADIR=../data/versioned_data/


# HM3D scenes
DATASET="hm3d"
DATASET_SPLIT="minival"
# SCENES=("00800-TEEsavR23oF" "00802-wcojb4TFT35")
SCENES=("00802-wcojb4TFT35")

# Gibson scenes
# DATASET="gibson"
# DATASET_SPLIT="val"
# # SCENES=("Greigsville" "Denmark" "Cantwell" "Eudora" "Pablo" "Ribera" "Swormville" "Eastville" "Elmira")
# SCENES=("Greigsville")


# MP3D scenes
# DATASET="MP3D"
# DATASET_SPLIT="train"
# # SCENES=("GdvgFV5R1Z5" "gZ6f7yhEvPG" "HxpKQynjfin" "pLe4wQe7qrG" "YmJkqBEsHnH")
# SCENES=("gZ6f7yhEvPG")

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
                    --dataset ${DATASET} \
                    --dataset_split ${DATASET_SPLIT} 
done