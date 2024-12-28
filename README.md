# FisherRF Active Mapping Part

## Prepare data

We use Gibson and HM3D dataset. 

The gibson dataset can be downloaded from [here](https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform). We use scenes `Greigsville` `Denmark` `Cantwell` `Eudora` `Pablo` `Ribera` `Swormville` `Eastville` `Elmira`.

The HM3D dataset can be downloaded from [here](https://niessner.github.io/Matterport/#download). We use scenes `DBjEcHFg4oq` `mscxX4KEBcB` `QKGMrurUVbk` `oPj9qMxrDEa` `CETmJJqkhcK` .

The data should be organized as 

```
-- habitat-api
    |
    - - scene_datasets
        |
        - - hm3d
        |   |
        |   - - DBjEcHFg4oq
        |   ...
        |
        - - gibson
            |
            - - Denmark
            ...
```

## Setup 

We highly recommend using the docker image

```bash
# pull image
docker pull wen3d/agslam:latest

# run docker image
docker run -it --runtime=nvidia \
		-e QT_X11_NO_MITSHM=1  \
		-e NVIDIA_VISIBLE_DEVICES=all \
		-e NVIDIA_DRIVER_CAPABILITIES=all  \
		--cpus=16 --memory=48g --shm-size=16g \
		-v /home/kostas-lab/Documents/release:/root \
        -v /home/kostas-lab/data:/data \
		--cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
		-p 4461:80 -p 4462:5900 -p 4463:22 \
		-e VNC_PASSWORD=YOUR_PASSWORD -e HTTP_PASSWORD=YOUR_PASSWORD \
		wen3d/agslam:latest
```

For GUI access, please refer to this [repo](https://github.com/fcwu/docker-ubuntu-vnc-desktop).

You can access the vnc in your browser through `localhost:4461`

### Compile from source

We use habitat-sim and habitat-lab (both are v0.2.4). Please refer to the documentation [here](https://github.com/facebookresearch/habitat-sim/blob/f179b584bcd713c5a2a998132211e2cae881d6d1/BUILD_FROM_SOURCE.md).

## Run Experiment

```bash
# In your docker container, clone the repo
git clone xxx --recursive

cd thirdparty/simple-knn && python -m pip install -e . 
cd thirdparty/diff-gaussian-rasterization-modified && python -m pip install -e . 

# The dataset dir can be specified by the `DATADIR` variable

# FisherRF results
bash scripts/gibson.sh configs/mp3d_gaussian_FR_eccv.yaml

# Frontier
bash scripts/gibson.sh configs/mp3d_gaussian_FR_frontier.yaml

# UPEN
bash scripts/gibson.sh configs/mp3d_gaussian_UPEN_fbe.yaml

# HM3D results
bash scripts/mp3d.sh configs/mp3d_gaussian_FR_eccv.yaml
```

Pretrained Gaussians can be found [here](https://drive.google.com/drive/folders/15aTH4025cbjs1Y81g1PjKbdSXOixgboV?usp=sharing).
