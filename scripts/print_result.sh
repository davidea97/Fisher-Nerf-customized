BASE_DIR=eccv_reproduce

SCENES=("Greigsville" "Denmark" "Cantwell" "Eudora" "Pablo" "Ribera" "Swormville" "Eastville" "Elmira" )

for SCENE in ${SCENES[@]};
do
    cat experiments/GaussianSLAM/${SCENE}-${BASE_DIR}/results.txt
done