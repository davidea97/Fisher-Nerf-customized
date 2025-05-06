# FisherRF Active Mapping Part





## Prepare data

The data should be organized as 

```
-- data
    |
    - - versioned_data
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

# 🧭 Frontier-Based Navigation with Habitat

This project provides a framework for **frontier-based active exploration** using Habitat and a simplified navigation pipeline (without actual Gaussian splatting).

---

## 🚀 How to Run

The main entry point is the script `mp3d.sh`. You need to:

1. **Set the `DATADIR`** variable to point to your `versioned_data` folder.
2. **List the scenes** you want to process (see examples inside `mp3d.sh`).

This will launch the main Python script:

```bash
python main_navigation.py
```

according to the configuration specified in:

```
configs/mp3d_gaussian_FR_eccv.yaml
```

---

## ⚙️ Configuration Notes

- The **camera intrinsics** are configured based on the `img_size` parameter.
- **⚠️ Warning:** The original authors hardcoded some parameters based on `img_size`. Currently, the code assumes `img_size=256x256`. Changing this value may require updating other parts of the code accordingly.

---

## 🧠 Pipeline Breakdown

1. Inside `main_navigation.py`, you need to set:
   - `dataset_type` (e.g., `hm3d`)
   - The evaluation `split`

2. A `Navigator` object is created from `tester_navigator.py`, which runs frontier-based navigation.

3. The **Habitat environment** is initialized using settings defined in `train_options.py`.

4. A `GaussianSLAM` object is also created:
   - **Note:** This is used **only for loading configuration files**. Gaussian splats are **not used** in this navigation pipeline.

---

## 🔄 Exploration Logic

- The agent begins by performing a **360-degree scan** of the environment.
- At each step, the **occupancy map** is updated based on camera depth measurements.

---

## 🌍 Frontier Planning

- The global planning module in `astar.py` builds **frontiers** from the current occupancy map.
- It generates **6D candidate poses** using the `"combined"` policy (a trade-off between selecting the largest and the closest frontier).
   - This behavior can be changed in `mp3d_gaussian_FR_eccv.yaml`.

- Then, `action_planning_frontier()` uses the **A\\*** algorithm to plan a path toward the selected frontier.

---

## 📦 Other Functions

- `store_filtered_pointcloud()`:
   - Updates the global point cloud at every step.
   - By default, **only 5%** of the new points are retained (can be changed manually).

- `count_visible_points()`:
   - Takes as input:
     - A global point cloud (`numpy` array),
     - A 6D camera pose,
     - The camera intrinsics,
     - The image size.
   - Returns the number of 3D points visible from that pose.



```bash
# In your env, clone the repo
git clone xxx --recursive

cd thirdparty/simple-knn && python -m pip install -e . 
cd thirdparty/diff-gaussian-rasterization-modified && python -m pip install -e . 

# FisherRF results frontier
bash scripts/mp3d.sh configs/mp3d_gaussian_FR_eccv.yaml
```