
CONFIG=$1
DATADIR=/data/habitat-api

SCENES=("Ribera" )
# "Cantwell" "Eudora" "Pablo" "Ribera" "Greigsville"
for scene in ${SCENES[@]}
do
    python main.py --name test_pointnav_exp \
                    --ensemble_dir ckpt/ \
                    --slam_config ${CONFIG} \
                    --root_path ${DATADIR} \
                    --log_dir logs/ \
                    --scenes_list ${scene} \
                    --gpu_capacity 1 \
                    --test_set v1  \
                    --dataset_type gibson \
                    --scenes_dir gibson/ 
done