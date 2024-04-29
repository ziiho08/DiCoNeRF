# DiCo-NeRF
We integrate CLIP embeddings into the NeRF optimization process, which allows us to leverage semantic information provided by CLIP when synthesizing novel views of fisheye driving scenes. The proposed method, DiCo-NeRF, utilizes the distributional differences between the similarity maps obtained from pre-trained CLIP to improve the color field of the NeRF.

![Fig1](https://github.com/ziiho08/DiCoNeRF/assets/68531659/e25f9d3c-c4b7-4aa0-8d13-65c63d2214ec)

## Installation
Ensure that nerfstudio has been installed according to the [instructions](https://docs.nerf.studio/quickstart/installation.html). 
Then, clone this repository and run the commands:
```
git clone https://github.com/ziiho08/DiCoNeRF.git
conda activate nerfstudio
cd diconerf/
pip install -e .
ns-install-cli
```

## Training the DiCo-NeRF
To train the DiCo-NeRF, run the command:
```
ns-train diconerf --data [PATH]
```

## Demo
<img width="250" height="250" src="!https://github.com/ziiho08/DiCoNeRF/assets/68531659/763b9fad-3038-40bb-8810-a18cc582a1cc"/>
<img width="250" height="250" src="https://github.com/ziiho08/DiCoNeRF/assets/68531659/23b6192c-c366-4fbe-b928-48a54ff9f141"/>
<img width="350" height="250" src="https://github.com/ziiho08/DiCoNeRF/assets/68531659/fb7900ca-51e5-4b1d-a594-2ae02acfa9b7"/>

## Citation
```

```
